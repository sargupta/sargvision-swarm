"""Live frame renderer — Pillow 2D top-down view of a swarm in flight.

Designed for ~10 fps streaming through Gradio. Each call returns a PIL.Image you
can hand straight to gr.Image as numpy.

Visual language:
- Drone        = filled circle, role-colored, size proportional to z.
- Trail        = faded line of last 12 positions per drone.
- Comm edge    = thin translucent line drone↔drone if within comm range.
- Message      = bright arrow that fades over 4 frames, color-keyed by protocol.
- Intent label = tiny text under each drone showing the LLM-emitted intent.
- Event tag    = floating annotation near event location for ~6 frames.
- BFT flash    = the 7 committee members briefly outlined yellow.
- CBBA paint   = task cell square flashes when claimed.
"""

from __future__ import annotations

import io
import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from sargvision_swarm.comms.protocols import Protocol
from sargvision_swarm.core.state import Role, SwarmState

# ── Layout ──────────────────────────────────────────────────────────────

CANVAS_W = 1000
CANVAS_H = 700
TITLE_H = 64
LEGEND_H = 40
WORLD_X_RANGE = (-30.0, 30.0)  # meters
WORLD_Y_RANGE = (-30.0, 30.0)


# ── Colors ──────────────────────────────────────────────────────────────

BG = (12, 18, 28)
GRID = (35, 45, 60)
TITLE_BG = (8, 14, 22)
TEXT = (235, 240, 245)
TEXT_DIM = (135, 145, 160)
TEXT_VDIM = (90, 100, 115)

ROLE_COLOR = {
    Role.WORKER: (96, 165, 250),
    Role.SCOUT: (52, 211, 153),
    Role.RELAY: (251, 191, 36),
    Role.LEADER: (248, 113, 113),
}

PROTOCOL_COLOR = {
    Protocol.A2A: (167, 139, 250),  # purple
    Protocol.MCP: (244, 114, 182),  # pink
    Protocol.MAVLINK: (52, 211, 153),  # green
    Protocol.ZENOH: (96, 165, 250),  # blue
    Protocol.DDS: (96, 165, 250),
    Protocol.GRPC: (251, 146, 60),  # orange
    Protocol.BFT: (248, 113, 113),  # red
}

EDGE_COLOR = (16, 185, 129, 50)  # rgba dimmed green
TRAIL_BASE = (96, 165, 250)


# ── Helpers ─────────────────────────────────────────────────────────────


def _world_to_px(x: float, y: float) -> tuple[int, int]:
    """Map (x, y) world coords → pixel coords inside the canvas main area."""
    wx0, wx1 = WORLD_X_RANGE
    wy0, wy1 = WORLD_Y_RANGE
    main_top = TITLE_H
    main_h = CANVAS_H - TITLE_H - LEGEND_H
    px = int((x - wx0) / (wx1 - wx0) * CANVAS_W)
    py = int(main_top + (1.0 - (y - wy0) / (wy1 - wy0)) * main_h)
    return px, py


_FONT_TITLE: ImageFont.FreeTypeFont | ImageFont.ImageFont | None = None
_FONT_BODY: ImageFont.FreeTypeFont | ImageFont.ImageFont | None = None
_FONT_TINY: ImageFont.FreeTypeFont | ImageFont.ImageFont | None = None


def _font(size: int):
    """Load a system font; fall back to default if not found."""
    candidates = [
        "/System/Library/Fonts/SFNS.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Geneva.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _init_fonts() -> None:
    global _FONT_TITLE, _FONT_BODY, _FONT_TINY
    if _FONT_TITLE is None:
        _FONT_TITLE = _font(20)
        _FONT_BODY = _font(13)
        _FONT_TINY = _font(10)


# ── Event + recent-message buffers ──────────────────────────────────────


@dataclass
class TrailHistory:
    capacity: int = 12
    _by_drone: dict[int, deque[tuple[float, float]]] = field(default_factory=dict)

    def push(self, drone_id: int, x: float, y: float) -> None:
        if drone_id not in self._by_drone:
            self._by_drone[drone_id] = deque(maxlen=self.capacity)
        self._by_drone[drone_id].append((x, y))

    def get(self, drone_id: int) -> list[tuple[float, float]]:
        return list(self._by_drone.get(drone_id, []))


@dataclass
class RecentMessage:
    src_id: int
    dst_id: int | None
    protocol: Protocol
    age: int = 0
    src_xy: tuple[float, float] = (0.0, 0.0)
    dst_xy: tuple[float, float] | None = None


@dataclass
class FloatingEvent:
    text: str
    x: float
    y: float
    color: tuple[int, int, int] = (255, 255, 255)
    age: int = 0


# ── Main render ─────────────────────────────────────────────────────────


def render_frame(
    swarm: SwarmState,
    trails: TrailHistory,
    recent_msgs: list[RecentMessage],
    floating_events: list[FloatingEvent],
    comm_adjacency: np.ndarray,
    intents: dict[int, str],
    stats: dict,
    bft_flash_voters: set[int] | None = None,
    cbba_flash_cells: dict[str, tuple[float, float]] | None = None,
) -> Image.Image:
    """Render one frame as a PIL Image."""
    _init_fonts()
    img = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)
    draw = ImageDraw.Draw(img, "RGBA")

    # ── Title bar ──
    draw.rectangle((0, 0, CANVAS_W, TITLE_H), fill=TITLE_BG)
    title = "SARGVISION Swarm — LIVE"
    draw.text((16, 8), title, fill=TEXT, font=_FONT_TITLE)
    sub = (
        f"t={stats.get('t', 0):.2f}s  ·  N={swarm.n}  ·  "
        f"msgs total={stats.get('total_msgs', 0)}  "
        f"({stats.get('msgs_per_s', 0):.0f}/s)  ·  "
        f"BFT votes={stats.get('bft_count', 0)}  ·  "
        f"CBBA bids={stats.get('cbba_count', 0)}"
    )
    draw.text((16, 36), sub, fill=TEXT_DIM, font=_FONT_BODY)

    # ── Grid ──
    main_top = TITLE_H
    main_bottom = CANVAS_H - LEGEND_H
    for gx in range(-30, 31, 10):
        x0, _ = _world_to_px(gx, 0)
        draw.line([(x0, main_top), (x0, main_bottom)], fill=GRID, width=1)
    for gy in range(-30, 31, 10):
        _, y0 = _world_to_px(0, gy)
        draw.line([(0, y0), (CANVAS_W, y0)], fill=GRID, width=1)

    positions = swarm.positions

    # ── CBBA cell paints (under everything) ──
    if cbba_flash_cells:
        for _, (cx, cy) in cbba_flash_cells.items():
            px, py = _world_to_px(cx, cy)
            draw.rectangle((px - 12, py - 12, px + 12, py + 12), outline=(251, 191, 36), width=2)

    # ── Comm-range edges ──
    n = swarm.n
    for i in range(n):
        for j in range(i + 1, n):
            if comm_adjacency[i, j]:
                x1, y1 = _world_to_px(positions[i, 0], positions[i, 1])
                x2, y2 = _world_to_px(positions[j, 0], positions[j, 1])
                draw.line([(x1, y1), (x2, y2)], fill=EDGE_COLOR, width=1)

    # ── Drone trails ──
    for d in swarm.drones:
        pts = trails.get(d.id)
        if len(pts) < 2:
            continue
        for i in range(1, len(pts)):
            x1, y1 = _world_to_px(*pts[i - 1])
            x2, y2 = _world_to_px(*pts[i])
            alpha = int(180 * (i / len(pts)))
            draw.line([(x1, y1), (x2, y2)], fill=(*TRAIL_BASE, alpha), width=1)

    # ── Recent message arrows (fading) ──
    for m in recent_msgs:
        color = PROTOCOL_COLOR.get(m.protocol, (200, 200, 200))
        max_age = 5
        if m.age >= max_age:
            continue
        alpha = int(220 * (1.0 - m.age / max_age))
        x1, y1 = _world_to_px(*m.src_xy)
        if m.dst_xy is None:
            # broadcast — a small ring around source
            r = 8 + m.age * 4
            draw.ellipse(
                (x1 - r, y1 - r, x1 + r, y1 + r),
                outline=(*color, alpha),
                width=2,
            )
        else:
            x2, y2 = _world_to_px(*m.dst_xy)
            _draw_arrow(draw, x1, y1, x2, y2, (*color, alpha), width=2 + (m.age == 0))

    # ── Drones ──
    for i, d in enumerate(swarm.drones):
        x, y = _world_to_px(d.pos[0], d.pos[1])
        color = ROLE_COLOR.get(d.role, ROLE_COLOR[Role.WORKER])
        z_scale = max(0.6, min(1.6, d.pos[2] / 6.0))
        r = int(7 * z_scale)
        # Halo if Byzantine flash or BFT voter
        if bft_flash_voters and d.id in bft_flash_voters:
            draw.ellipse(
                (x - r - 5, y - r - 5, x + r + 5, y + r + 5), outline=(252, 211, 77), width=2
            )
        # Disc
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color)
        # ID
        draw.text((x + r + 3, y - 6), str(d.id), fill=TEXT, font=_FONT_TINY)
        # Intent label below
        intent = intents.get(d.id, "")
        if intent:
            label = _short_intent(intent)
            draw.text((x - 20, y + r + 2), label, fill=TEXT_DIM, font=_FONT_TINY)

    # ── Floating event tags ──
    for ev in floating_events:
        if ev.age > 8:
            continue
        x, y = _world_to_px(ev.x, ev.y)
        alpha = int(240 * (1.0 - ev.age / 8))
        # background pill
        bbox = draw.textbbox((x, y), ev.text, font=_FONT_BODY)
        w_text = bbox[2] - bbox[0]
        h_text = bbox[3] - bbox[1]
        draw.rectangle(
            (x - 4, y - 14, x + w_text + 4, y + h_text - 10),
            fill=(0, 0, 0, 180),
        )
        draw.text((x, y - 12), ev.text, fill=(*ev.color, alpha), font=_FONT_BODY)

    # ── Legend / protocol counts ──
    draw.rectangle((0, main_bottom, CANVAS_W, CANVAS_H), fill=TITLE_BG)
    proto_counts: dict[str, int] = stats.get("by_proto", {})
    legend_x = 16
    legend_y = main_bottom + 12
    for proto in [Protocol.A2A, Protocol.ZENOH, Protocol.MAVLINK, Protocol.BFT, Protocol.GRPC]:
        c = PROTOCOL_COLOR[proto]
        draw.rectangle((legend_x, legend_y + 2, legend_x + 14, legend_y + 14), fill=c)
        count = proto_counts.get(proto.value, 0)
        label = f"{proto.value} {count}"
        draw.text((legend_x + 18, legend_y), label, fill=TEXT, font=_FONT_BODY)
        legend_x += 130

    return img


def _draw_arrow(draw, x1, y1, x2, y2, color, width=2):
    """Line with arrowhead at (x2, y2)."""
    draw.line([(x1, y1), (x2, y2)], fill=color, width=width)
    dx, dy = x2 - x1, y2 - y1
    L = math.sqrt(dx * dx + dy * dy)
    if L < 4:
        return
    ux, uy = dx / L, dy / L
    # arrowhead
    head_len = 8
    px, py = -uy, ux
    hx = x2 - ux * head_len
    hy = y2 - uy * head_len
    draw.polygon(
        [
            (x2, y2),
            (hx + px * 4, hy + py * 4),
            (hx - px * 4, hy - py * 4),
        ],
        fill=color,
    )


_INTENT_SHORT = {
    "hold_formation": "HOLD",
    "advance_to_goal": "ADV",
    "yield_to_neighbor": "YLD",
    "rotate_role": "ROT",
    "report_health": "HB",
}


def _short_intent(intent: str) -> str:
    return _INTENT_SHORT.get(intent, intent[:4].upper())


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def to_numpy(img: Image.Image) -> np.ndarray:
    return np.asarray(img)
