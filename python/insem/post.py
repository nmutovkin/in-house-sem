"""Post-processing: results files, ParaView (XDMF) snapshots and SEG-Y export."""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union
from xml.sax.saxutils import escape

import numpy as np


# ------------------------------------------------------------------ results


@dataclass
class Receiver:
    name: str
    x: float
    z: float
    medium: str


@dataclass
class Results:
    """Contents of a results file written by ``Simulation.write_results``."""

    dt: float
    time: np.ndarray
    receivers: Dict[str, Receiver]
    #: (receiver, quantity) -> (n_samples, ncomp)
    traces: Dict[Tuple[str, str], np.ndarray]
    #: (n, 3): time, kinetic, potential (empty without energy monitoring)
    energy: np.ndarray
    snapshot_time: np.ndarray = field(default_factory=lambda: np.zeros(0))
    #: medium -> (points (n, 2), cells (m, 4))
    snapshot_mesh: Dict[str, Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    #: (medium, quantity) -> (n_snapshots, ncomp, n_points)
    snapshots: Dict[Tuple[str, str], np.ndarray] = field(default_factory=dict)

    def trace(self, receiver: str, quantity: str) -> np.ndarray:
        return self.traces[(receiver, quantity)]


def _text(v) -> str:
    return v.decode() if isinstance(v, bytes) else str(v)


def read_results(path) -> Results:
    import h5py

    with h5py.File(path, "r") as f:
        if _text(f.attrs.get("format", "")) != "insem-results":
            raise ValueError(f"{path} is not an insem results file")
        receivers, traces = {}, {}
        for name, group in f.get("receivers", {}).items():
            receivers[name] = Receiver(name, float(group.attrs["x"]), float(group.attrs["z"]),
                                       _text(group.attrs["medium"]))
            for quantity, data in group.items():
                traces[(name, quantity)] = data[()]
        res = Results(
            dt=float(f.attrs["dt"]),
            time=f["time"][()],
            receivers=receivers,
            traces=traces,
            energy=f["energy"][()] if "energy" in f else np.zeros((0, 3)),
        )
        if "snapshots" in f:
            snap = f["snapshots"]
            res.snapshot_time = snap["time"][()]
            for medium, group in snap.items():
                if medium == "time":
                    continue
                res.snapshot_mesh[medium] = (group["points"][()], group["cells"][()])
                for quantity, data in group.items():
                    if quantity not in ("points", "cells"):
                        res.snapshots[(medium, quantity)] = data[()]
    return res


# ------------------------------------------------------------------- XDMF

_COMPONENTS = {1: [""], 2: ["_x", "_z"]}


def write_xdmf(results_path, xdmf_path=None) -> Path:
    """Write an XDMF file for the wavefield snapshots of a results file.

    Opens in ParaView (Xdmf3 reader) as a time series with one block per
    medium; vector quantities appear as _x / _z components. The data is
    written to a sidecar HDF5 file next to the XDMF file.
    """
    import h5py

    results_path = Path(results_path)
    res = read_results(results_path)
    if not res.snapshots:
        raise ValueError(f"{results_path} has no snapshots (set snapshot_every in set_time)")
    xdmf_path = Path(xdmf_path) if xdmf_path else results_path.with_suffix(".xdmf")
    side = xdmf_path.with_suffix(".xdmf.h5")
    with h5py.File(side, "w") as h:
        for medium, (points, cells) in res.snapshot_mesh.items():
            h[f"{medium}/points"] = points
            h[f"{medium}/cells"] = cells
        for (medium, quantity), data in res.snapshots.items():
            for c, suffix in enumerate(_COMPONENTS[data.shape[1]]):
                for s in range(data.shape[0]):
                    h[f"{medium}/{quantity}{suffix}/{s}"] = data[s, c]

    ref = escape(side.name)
    lines = ['<?xml version="1.0" ?>', '<Xdmf Version="3.0">', "  <Domain>",
             '    <Grid Name="snapshots" GridType="Collection" CollectionType="Temporal">']
    for s, t in enumerate(res.snapshot_time):
        lines.append(f'      <Grid Name="step{s}" GridType="Collection" CollectionType="Spatial">')
        lines.append(f'        <Time Value="{t:.9g}"/>')
        for medium, (points, cells) in res.snapshot_mesh.items():
            n, m = len(points), len(cells)
            lines += [
                f'        <Grid Name="{medium}" GridType="Uniform">',
                f'          <Topology TopologyType="Quadrilateral" NumberOfElements="{m}">',
                f'            <DataItem Dimensions="{m} 4" NumberType="Int" Precision="4" Format="HDF">'
                f"{ref}:/{medium}/cells</DataItem>",
                "          </Topology>",
                '          <Geometry GeometryType="XY">',
                f'            <DataItem Dimensions="{n} 2" NumberType="Float" Precision="8" Format="HDF">'
                f"{ref}:/{medium}/points</DataItem>",
                "          </Geometry>",
            ]
            for (med, quantity), data in res.snapshots.items():
                if med != medium:
                    continue
                for suffix in _COMPONENTS[data.shape[1]]:
                    lines += [
                        f'          <Attribute Name="{quantity}{suffix}" AttributeType="Scalar" Center="Node">',
                        f'            <DataItem Dimensions="{n}" NumberType="Float" Precision="4" Format="HDF">'
                        f"{ref}:/{medium}/{quantity}{suffix}/{s}</DataItem>",
                        "          </Attribute>",
                    ]
            lines.append("        </Grid>")
        lines.append("      </Grid>")
    lines += ["    </Grid>", "  </Domain>", "</Xdmf>", ""]
    xdmf_path.write_text("\n".join(lines))
    return xdmf_path


# ------------------------------------------------------------------- SEG-Y

_COORD_SCALAR = -1000  # coordinates stored in millimetres


def write_segy(
    results_path,
    segy_path,
    quantity: str,
    component: Union[int, str] = 0,
    receivers: Optional[Sequence[str]] = None,
    source: Optional[Tuple[float, float]] = None,
    text: str = "",
) -> Path:
    """Write one component of a recorded quantity as a SEG-Y rev 2 file.

    One trace per receiver (in the given order, or file order), IEEE float
    samples, big-endian. Receiver x goes to the group X coordinate and z to
    the group elevation, both in millimetres (scalar -1000); `source`
    (x, z) fills the source coordinates.

    The exact sample interval goes to the rev 2 extended field (IEEE double,
    microseconds); the classic 16-bit field holds it rounded to whole
    microseconds, which is what readers that ignore rev 2 extensions see.
    For such readers choose `record_every` so that the sample interval is a
    whole number of microseconds, or resample.
    """
    res = read_results(results_path)
    names = list(receivers) if receivers is not None else [n for n in res.receivers
                                                          if (n, quantity) in res.traces]
    if not names:
        raise ValueError(f"no receiver recorded '{quantity}'")
    comp = {"x": 0, "z": 1}.get(component, component) if isinstance(component, str) else component
    data = []
    for n in names:
        tr = res.traces.get((n, quantity))
        if tr is None:
            raise ValueError(f"receiver '{n}' did not record '{quantity}'")
        if comp >= tr.shape[1]:
            raise ValueError(f"'{quantity}' has {tr.shape[1]} component(s)")
        data.append(tr[:, comp])
    data = np.asarray(data, dtype=">f4")
    ns = data.shape[1]
    if ns > 32767:
        raise ValueError("more than 32767 samples per trace; increase record_every")
    dt = float(res.time[1] - res.time[0]) if len(res.time) > 1 else res.dt
    dt_us = dt * 1e6

    lines = [f"C{i + 1:2d} " for i in range(40)]
    body = ["insem synthetic seismograms", f"quantity {quantity} component {component}",
            f"{len(names)} traces, {ns} samples, dt = {dt:.9g} s",
            "coordinates in mm (scalar -1000): x = group X, z = group elevation"]
    body += [t for t in text.splitlines()]
    for i, t in enumerate(body[:38]):
        lines[i] += t
    lines[38] += "SEG-Y REV2.0"
    lines[39] += "END TEXTUAL HEADER"
    textual = "".join(line[:80].ljust(80) for line in lines).encode("cp037")

    binary = bytearray(400)

    def put(buf, offset, fmt, value):  # offset: 1-based byte position within the header
        struct.pack_into(">" + fmt, buf, offset - 1, value)

    put(binary, 1, "i", 1)
    put(binary, 5, "i", 1)
    put(binary, 9, "i", 1)
    put(binary, 13, "h", min(len(names), 32767))
    put(binary, 17, "h", min(int(round(dt_us)), 32767))
    put(binary, 19, "h", min(int(round(dt_us)), 32767))
    put(binary, 21, "h", ns)
    put(binary, 23, "h", ns)
    put(binary, 25, "h", 5)       # IEEE float
    put(binary, 27, "h", 1)       # ensemble fold
    put(binary, 29, "h", 1)       # sorting: as recorded
    put(binary, 55, "h", 1)       # metres
    # SEG-Y rev 2 extended fields (bytes 3261-3296).
    put(binary, 61, "i", len(names))  # traces per ensemble
    put(binary, 69, "i", ns)          # samples per trace
    put(binary, 73, "d", dt_us)       # sample interval (us)
    put(binary, 81, "d", dt_us)       # original sample interval
    put(binary, 89, "i", ns)          # original samples per trace
    put(binary, 93, "i", 1)           # ensemble fold
    put(binary, 97, "I", 0x01020304)  # byte-order constant
    binary[300] = 2               # revision 2.0
    binary[301] = 0
    put(binary, 303, "h", 1)      # fixed trace length
    put(binary, 305, "h", 0)      # no extended textual headers

    def scaled(v):
        return int(round(v * -_COORD_SCALAR))

    out = bytearray(textual + bytes(binary))
    for i, n in enumerate(names):
        rec = res.receivers[n]
        h = bytearray(240)
        put(h, 1, "i", i + 1)
        put(h, 5, "i", i + 1)
        put(h, 9, "i", 1)
        put(h, 13, "i", i + 1)
        put(h, 29, "h", 1)
        put(h, 41, "i", scaled(rec.z))
        put(h, 69, "h", _COORD_SCALAR)
        put(h, 71, "h", _COORD_SCALAR)
        if source is not None:
            put(h, 45, "i", scaled(source[1]))
            put(h, 73, "i", scaled(source[0]))
            put(h, 37, "i", int(round(np.hypot(rec.x - source[0], rec.z - source[1]) * 1000)))
        put(h, 81, "i", scaled(rec.x))
        put(h, 89, "h", 1)
        put(h, 115, "h", ns)
        put(h, 117, "h", min(int(round(dt_us)), 32767))
        out += h + data[i].tobytes()
    segy_path = Path(segy_path)
    segy_path.write_bytes(bytes(out))
    return segy_path


@dataclass
class SegyData:
    text: str
    dt: float
    #: (n_traces, n_samples)
    data: np.ndarray
    group_x: np.ndarray
    group_z: np.ndarray
    source_x: np.ndarray
    source_z: np.ndarray


def read_segy(path) -> SegyData:
    """Read a fixed-length SEG-Y file with IEEE float samples (as written by write_segy)."""
    raw = Path(path).read_bytes()
    text = raw[:3200].decode("cp037")
    b = raw[3200:3600]

    def get(buf, offset, fmt):
        return struct.unpack_from(">" + fmt, buf, offset - 1)[0]

    if get(b, 25, "h") != 5:
        raise ValueError("only IEEE float (format 5) SEG-Y files are supported")
    ns = get(b, 21, "h")
    dt_us = get(b, 73, "d") or float(get(b, 17, "h"))
    size = 240 + 4 * ns
    n = (len(raw) - 3600) // size
    data = np.empty((n, ns), dtype=np.float32)
    gx, gz, sx, sz = (np.empty(n) for _ in range(4))
    for i in range(n):
        h = raw[3600 + i * size:3600 + i * size + 240]
        scalar = get(h, 71, "h")
        zscalar = get(h, 69, "h")

        def unscale(v, s):
            return v / -s if s < 0 else v * (s or 1)

        gx[i] = unscale(get(h, 81, "i"), scalar)
        gz[i] = unscale(get(h, 41, "i"), zscalar)
        sx[i] = unscale(get(h, 73, "i"), scalar)
        sz[i] = unscale(get(h, 45, "i"), zscalar)
        data[i] = np.frombuffer(raw, dtype=">f4", count=ns, offset=3600 + i * size + 240)
    return SegyData(text, dt_us * 1e-6, data, gx, gz, sx, sz)
