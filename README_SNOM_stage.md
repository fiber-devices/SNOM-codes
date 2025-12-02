# Fiber Alignment System — Multi-Controller Integration Framework
**Cylindrical Coordinates: x = horizontal, y = vertical, z = fiber axis**  
**Software Version:** `SNOM_stage.py`

---

## 🧩 Overview
This system provides a unified, programmable framework for controlling six high-precision motion axes used in SNOM (Scanning Near-field Optical Microscopy) and nanofiber alignment.  
It interfaces directly with **Thorlabs Kinesis .NET API** via `pythonnet`, combining DC servo, piezo, and encoded stages into one coordinate system.

---

## ⚙️ Hardware Configuration
| Axis | Stage / Actuator | Controller | Mode | Description |
|------|------------------|-------------|------|--------------|
| **Pitch** | Rotation stage | KDC101 #1 | Closed-loop | Angular alignment about x-axis |
| **Yaw** | Rotation stage | KDC101 #2 | Closed-loop | Angular alignment about y-axis |
| **Y (coarse)** | Linear DC stage | KDC101 #3 | Closed-loop | Vertical coarse translation |
| **Y (fine)** | LPS710E/M | Integrated | Open / Closed-loop | Vertical fine translation (~1 mm travel) |
| **Z (coarse)** | PDX1/M ORIC stage | PDXC2 | Closed-loop | Fiber-axis motion (10–20 mm) |
| **X (fine)** | NF15AP25/M Flexure Piezo | KPC101 | Closed-loop | Horizontal fine motion (25 µm, sub-nm precision) |

> 🛈 Note: `KIM001` (Z fine inertia motor) has been removed — it is not suitable for position-based feedback control.

---

## 🧠 Controller Compatibility
| Controller | Drives | Loop Type | Typical Stage |
|-------------|---------|-----------|----------------|
| **KDC101** | DC servo | Closed-loop | PRM1Z8 / Z9 / MT1 |
| **KPC101** | Piezo stack | Open / Closed-loop | NF15AP25/M |
| **PDXC2** | Encoder piezo | Closed-loop | PDX1(/M), PDX2(/M) |
| **LPS710E/M** | Integrated DC stage | Open / Closed-loop | Built-in encoder |
| **KEH6** | — | — | Power + communication hub |

---

## 🧱 Coordinate Definitions
| Symbol | Axis | Direction | Typical Travel |
|---------|------|------------|----------------|
| **x** | Horizontal | Radial (left–right) | 25 µm |
| **y** | Vertical | Up–down (fiber height) | 1 mm (fine), > 5 mm (coarse) |
| **z** | Fiber axis | Longitudinal | 10–20 mm |

---

## 🧩 Software Architecture
**Class:** `FiberHeadController`  
Defines unified motion interfaces for all axes using the Kinesis .NET API through `pythonnet`.

### Key Methods
| Method | Description |
|---------|-------------|
| `home_all()` | Homes all closed-loop axes (skips open-loop). |
| `rot(pitch, yaw)` | Rotate fiber mount about x/y. |
| `x(pos)` | Move X-axis piezo (µm). |
| `Y_coarse(pos)` | Move Y coarse stage (mm). |
| `y(pos)` | Move Y fine stage (µm). |
| `y_open_setup()` | Enable Y fine stage for open-loop analog control (external function generator). |
| `z(pos)` | Move Z coarse ORIC stage (µm). |
| `z_open_setup()` | Enable Z axis analog input for open-loop scanning. |
| `get_position(axis)` | Query current axis position. |
| `stop_all()` | Emergency stop for all controllers. |
| `close()` | Disconnect all devices safely. |

---

## ⚙️ Typical Serial Assignments
| Axis | Controller | Serial No. | Notes |
|------|-------------|-------------|--------|
| Pitch | KDC101 | 27000001 | Rotation |
| Yaw | KDC101 | 27000002 | Rotation |
| Y (coarse) | KDC101 | 27000003 | Vertical coarse |
| Y (fine) | LPS710E/M | 45000001 | Vertical fine |
| Z (coarse) | PDXC2 | 38000001 | Fiber axis |
| X (fine) | KPC101 | 29000001 | Horizontal piezo |

---

## 🧩 Example Workflow
```python
from SNOM_stage import FiberHeadController

with FiberHeadController() as ctrl:
    ctrl.home_all()  # Home closed-loop axes
    
    # Angular alignment
    ctrl.rot(pitch=0.02, yaw=-0.02)
    
    # Coarse motion
    ctrl.Y_coarse(5.0)   # mm vertical
    ctrl.z(10.0)         # mm fiber-axis
    
    # Fine positioning
    ctrl.y(500.0)        # μm vertical fine
    ctrl.x(12.5)         # μm horizontal fine
    
    # Check current positions
    for a in ['pitch', 'yaw', 'y_coarse', 'y_fine', 'z_coarse', 'x_fine']:
        print(a, ctrl.get_position(a))
    
    # Optional: external open-loop setup
    ctrl.y_open_setup()
    ctrl.z_open_setup()
```

---

## ⚙️ Resolution Notes
| Axis | Resolution / Increment | Control Mode |
|------|-------------------------|---------------|
| **X (fine)** | ~0.76 nm (theoretical)  /  2–3 nm (practical) | Closed-loop piezo |
| **Y (fine)** | ~0.1 µm (encoder)  /  0.5 µm practical | Closed-loop |
| **Z (coarse)** | ~0.1 µm (encoder) | Closed-loop |
| **Pitch / Yaw** | < 0.001° | Closed-loop |

---

## 🧱 External Trigger Integration
Both **Y (fine)** and **Z (coarse)** can operate in **open-loop analog input** mode for waveform scanning.

```python
ctrl.z_open_setup()
ctrl.y_open_setup()
```

Connect analog signal generators to the controller analog input ports:
- **PDXC2** (Z_coarse)
- **LPS710E/M** (Y_fine)

---

## 📚 References
- [Thorlabs KDC101 – DC Servo Controller](https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=1704)  
- [Thorlabs KPC101 – Piezo Controller](https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=7105)  
- [Thorlabs PDXC2 – ORIC Controller](https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=24569)  
- [Thorlabs LPS710E/M – Translation Stage](https://www.thorlabs.com/thorproduct.cfm?partnumber=LPS710E/M)  
- [Thorlabs NF15AP25/M – NanoFlex Flexure Stage](https://www.thorlabs.com/thorproduct.cfm?partnumber=NF15AP25/M)  
- [Thorlabs KEH6 – K-Cube Controller Hub](https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=2424)
