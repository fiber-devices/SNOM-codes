"""
Fiber Alignment System - Multi-Controller Integration
Cylindrical Coordinate System (x: horizontal, y: vertical, z: fiber axis)

Hardware Configuration:
1. Pitch – KDC101 #1 (rotation, closed-loop)
2. Yaw – KDC101 #2 (rotation, closed-loop)
3. Y_coarse – KDC101 #3 (DC servo, closed-loop, vertical coarse)
4. Y_fine – LPS710E/M (integrated, open/closed-loop, ~1mm travel)
5. Z_coarse – PDXC2 + PDX1/M ORIC (closed-loop, 10-20mm range)
6. X_fine – KPC101 + NF15AP25/M (closed-loop, 25μm range)

Note: KIM001 Z_fine removed - open-loop inertia motor not suitable for position control
"""

import clr
import sys
import time
from pathlib import Path
from typing import Optional, Dict

# Add Thorlabs Kinesis DLL path
KINESIS_PATH = r"C:\Program Files\Thorlabs\Kinesis"
sys.path.append(KINESIS_PATH)

# Load Kinesis .NET assemblies
clr.AddReference("Thorlabs.MotionControl.DeviceManagerCLI")
clr.AddReference("Thorlabs.MotionControl.GenericMotorCLI")
clr.AddReference("Thorlabs.MotionControl.KCube.DCServoCLI")
clr.AddReference("Thorlabs.MotionControl.KCube.PiezoCLI")
clr.AddReference("Thorlabs.MotionControl.KCube.InertialMotorCLI")
clr.AddReference("Thorlabs.MotionControl.Benchtop.PiezoCLI")
clr.AddReference("Thorlabs.MotionControl.IntegratedStepperMotorsCLI")

from Thorlabs.MotionControl.DeviceManagerCLI import DeviceManagerCLI
from Thorlabs.MotionControl.GenericMotorCLI import MotorDirection
from Thorlabs.MotionControl.KCube.DCServoCLI import KCubeDCServo
from Thorlabs.MotionControl.KCube.PiezoCLI import KCubePiezo
from Thorlabs.MotionControl.KCube.InertialMotorCLI import KCubeInertialMotor
from Thorlabs.MotionControl.Benchtop.PiezoCLI import BenchtopPiezo
from Thorlabs.MotionControl.IntegratedStepperMotorsCLI import LongTravelStage
from System import Decimal


class FiberHeadController:
    """
    Unified controller for fiber alignment in cylindrical coordinates.
    
    Axes:
        - Pitch: KDC101 #1 (rotation, degrees)
        - Yaw: KDC101 #2 (rotation, degrees)
        - Y_coarse: KDC101 #3 (vertical DC servo, mm, closed-loop)
        - Y_fine: LPS710E/M (vertical fine, μm, open/closed-loop)
        - Z_coarse: PDXC2 (fiber axis, mm, closed-loop)
        - X_fine: KPC101 (horizontal, μm, closed-loop)
    """
    
    def __init__(self, serials: Optional[Dict[str, str]] = None):
        """
        Initialize all controllers.
        
        Args:
            serials: Dictionary mapping axis names to serial numbers.
        """
        # Default serial numbers (update with your actual serials)
        self.serials = serials or {
            'pitch': '27000001',      # KDC101 #1
            'yaw': '27000002',        # KDC101 #2
            'y_coarse': '27000003',   # KDC101 #3
            'y_fine': '45000001',     # LPS710E/M
            'z_coarse': '38000001',   # PDXC2
            'x_fine': '29000001',     # KPC101
        }
        
        # Device references
        self.devices = {}
        self.connected = False
        
        print("Initializing Fiber Head Controller...")
        self._build_device_list()
        self._connect_all()
        
    def _build_device_list(self):
        """Build list of connected devices."""
        DeviceManagerCLI.BuildDeviceList()
        print(f"Found {DeviceManagerCLI.GetDeviceListSize()} Thorlabs devices")
        
    def _connect_all(self):
        """Connect and initialize all controllers."""
        try:
            # Connect KDC101 controllers (Pitch, Yaw, Y_coarse)
            for axis in ['pitch', 'yaw', 'y_coarse']:
                self._connect_dc_servo(axis, self.serials[axis])
            
            # Connect LPS710E/M (Y_fine)
            self._connect_lps710(self.serials['y_fine'])
            
            # Connect PDXC2 (Z_coarse)
            self._connect_pdxc2(self.serials['z_coarse'])
            
            # Connect KPC101 (X_fine)
            self._connect_piezo(self.serials['x_fine'])
            
            self.connected = True
            print("✓ All controllers connected successfully")
            
        except Exception as e:
            print(f"✗ Connection error: {e}")
            self.close()
            raise
    
    def _connect_dc_servo(self, axis: str, serial: str):
        """Connect a KDC101 DC servo controller."""
        device = KCubeDCServo.CreateKCubeDCServo(serial)
        device.Connect(serial)
        time.sleep(0.5)
        device.StartPolling(250)
        time.sleep(0.5)
        device.EnableDevice()
        time.sleep(0.5)
        
        # Load motor configuration
        motor_config = device.LoadMotorConfiguration(serial)
        if axis == 'y_coarse':
            motor_config.DeviceSettingsName = "MT1/M-Z9"
        else:
            motor_config.DeviceSettingsName = "K10CR1"  # Pitch/Yaw rotation
        motor_config.UpdateCurrentConfiguration()
        
        device.SetSettings(device.MotorDeviceSettings, True, False)
        
        self.devices[axis] = device
        print(f"  ✓ {axis.upper()}: {serial} (KDC101)")
    
    def _connect_lps710(self, serial: str):
        """Connect LPS710E/M integrated stage."""
        device = LongTravelStage.CreateLongTravelStage(serial)
        device.Connect(serial)
        time.sleep(0.5)
        device.StartPolling(250)
        time.sleep(0.5)
        device.EnableDevice()
        time.sleep(0.5)
        
        self.devices['y_fine'] = device
        print(f"  ✓ Y_FINE: {serial} (LPS710E/M)")
    
    def _connect_pdxc2(self, serial: str):
        """Connect PDXC2 ORIC stage controller."""
        device = BenchtopPiezo.CreateBenchtopPiezo(serial)
        device.Connect(serial)
        time.sleep(0.5)
        
        # PDXC2 has channels - get channel 1
        channel = device.GetChannel(1)
        channel.StartPolling(250)
        time.sleep(0.5)
        channel.EnableDevice()
        time.sleep(0.5)
        
        self.devices['z_coarse'] = channel
        print(f"  ✓ Z_COARSE: {serial} (PDXC2)")
    
    def _connect_piezo(self, serial: str):
        """Connect KPC101 piezo controller."""
        device = KCubePiezo.CreateKCubePiezo(serial)
        device.Connect(serial)
        time.sleep(0.5)
        device.StartPolling(250)
        time.sleep(0.5)
        device.EnableDevice()
        time.sleep(0.5)
        
        self.devices['x_fine'] = device
        print(f"  ✓ X_FINE: {serial} (KPC101)")
    
    # ============ High-level motion commands ============
    
    def home_all(self):
        """Home all closed-loop axes (skip open-loop KIM001)."""
        print("Homing all axes...")
        for axis in ['pitch', 'yaw', 'y_coarse', 'y_fine', 'z_coarse']:
            if axis in self.devices:
                print(f"  Homing {axis}...")
                self.devices[axis].Home(60000)  # 60s timeout
        print("✓ Homing complete")
    
    # ---- Rotation Control ----
    
    def rot(self, pitch: Optional[float] = None, yaw: Optional[float] = None, wait: bool = True):
        """
        Move pitch and/or yaw rotation stages (degrees).
        
        Args:
            pitch: Pitch angle in degrees (None = don't move)
            yaw: Yaw angle in degrees (None = don't move)
            wait: Wait for motion to complete
        """
        if pitch is not None:
            self._move_dc_servo('pitch', pitch, wait=False)
        if yaw is not None:
            self._move_dc_servo('yaw', yaw, wait=False)
        if wait and (pitch is not None or yaw is not None):
            time.sleep(0.5)
    
    # ---- X Control (Horizontal) ----
    
    def x(self, pos: float, wait: bool = True):
        """
        Move X_fine piezo stage (μm).
        
        Args:
            pos: X position in μm (25μm range)
            wait: Wait for motion to complete
        """
        device = self.devices['x_fine']
        decimal_val = Decimal(float(pos))
        device.SetPosition(decimal_val)
        if wait:
            time.sleep(0.1)
    
    # ---- Y Control (Vertical) ----
    
    def Y_coarse(self, pos: float, wait: bool = True):
        """
        Move Y_coarse DC servo stage (mm).
        
        Args:
            pos: Y position in mm
            wait: Wait for motion to complete
        """
        self._move_dc_servo('y_coarse', pos, wait)
    
    def y(self, pos: float, wait: bool = True):
        """
        Move Y_fine in closed-loop mode (μm).
        
        Args:
            pos: Y position in μm (~1000μm travel)
            wait: Wait for motion to complete
        """
        device = self.devices['y_fine']
        decimal_val = Decimal(float(pos))
        device.MoveTo(decimal_val, 60000)
        if wait:
            time.sleep(0.1)
    
    def y_open_setup(self):
        """
        Setup Y_fine (LPS710E/M) for open-loop scanning with external trigger.
        
        Returns:
            Device reference for Y_fine
            
        Example:
            y_dev = ctrl.y_open_setup()
            # Connect function generator to LPS710E/M analog input
        """
        device = self.devices['y_fine']
        self.devices['y_open'] = device
        
        print(f"  ✓ Y_fine (LPS710E/M) ready for open-loop")
        print(f"  Note: Connect function generator to LPS710E/M analog input")
        
        return device
    
    # ---- Z Control (Fiber Axis) ----
    
    def z(self, pos: float, wait: bool = True):
        """
        Move Z_coarse ORIC stage (μm).
        
        Args:
            pos: Z position in μm (10,000-20,000μm range)
            wait: Wait for motion to complete
        """
        device = self.devices['z_coarse']
        decimal_val = Decimal(float(pos))
        device.SetPosition(decimal_val)
        if wait:
            time.sleep(0.3)
    
    def z_open_setup(self):
        """
        Setup Z_coarse (PDXC2) for open-loop scanning with external trigger.
        
        Returns:
            Device reference for Z_coarse
            
        Example:
            z_dev = ctrl.z_open_setup()
            # Connect function generator to PDXC2 analog input
        """
        device = self.devices['z_coarse']
        self.devices['z_open'] = device
        
        print(f"  ✓ Z_coarse (PDXC2) ready for open-loop")
        print(f"  Note: Connect function generator to PDXC2 analog input")
        
        return device
    
    def set_voltage(self, axis: str, voltage: float):
        """
        Drive a piezo axis by commanding a DC voltage (software-generated).
        
        Args:
            axis: Key of the device inside `self.devices`
                  (e.g., 'z_open', 'z_coarse', 'y_open', 'x_fine').
            voltage: Target voltage in volts (respect each controller's limit,
                     typically 0-75 V for KPC101/PDXC2/LPS710E/M).
        """
        device = self.devices.get(axis)
        if device is None:
            raise ValueError(f"Axis '{axis}' is not initialized.")
        
        # Some controllers expose the piezo functions on a nested PiezoDevice.
        piezo_device = getattr(device, "PiezoDevice", device)
        
        # Find any available voltage-setting method.
        setter = None
        for candidate in ("SetOutputVoltage", "SetVoltageOutput", "SetVoltage"):
            setter = getattr(piezo_device, candidate, None)
            if setter:
                break
        
        if setter is None:
            raise RuntimeError(
                f"Device for axis '{axis}' does not expose a voltage setter."
            )
        
        setter(Decimal(float(voltage)))
        print(f"  ✓ {axis} voltage set to {voltage:.3f} V (software-driven)")
    
    # ---- Utility Functions ----
    
    def _move_dc_servo(self, axis: str, position: float, wait: bool):
        """Generic DC servo movement."""
        device = self.devices[axis]
        decimal_val = Decimal(float(position))
        device.MoveTo(decimal_val, 60000)
        if wait:
            time.sleep(0.5)
    
    def get_position(self, axis: str) -> float:
        """Get current position of an axis."""
        device = self.devices.get(axis)
        if not device:
            raise ValueError(f"Unknown axis: {axis}")
        
        return float(device.Position)
    
    def stop_all(self):
        """Emergency stop all axes."""
        print("STOPPING ALL AXES")
        for axis, device in self.devices.items():
            try:
                if hasattr(device, 'Stop'):
                    device.Stop(60000)
            except:
                pass
    
    def close(self):
        """Disconnect all devices."""
        print("Closing connections...")
        for axis, device in self.devices.items():
            try:
                device.StopPolling()
                device.Disconnect()
                print(f"  ✓ {axis} disconnected")
            except Exception as e:
                print(f"  ✗ {axis} disconnect error: {e}")
        
        self.devices.clear()
        self.connected = False
        print("✓ All devices closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ============ Example Usage ============
if __name__ == "__main__":
    with FiberHeadController() as ctrl:
        # Home all axes
        ctrl.home_all()
        
        # Alignment sequence
        print("\nPerforming alignment...")
        
        # Angular alignment
        ctrl.rot(pitch=0.02, yaw=-0.02)
        
        # Coarse positioning
        ctrl.Y_coarse(5.0)      # 5mm vertical coarse
        ctrl.z(10.0)            # 10mm fiber axis
        
        # Fine positioning
        ctrl.y(500.0)           # 500μm vertical fine
        ctrl.x(12.5)            # 12.5μm horizontal
        
        # Check positions
        print(f"\nPositions:")
        print(f"  Pitch: {ctrl.get_position('pitch'):.4f}°")
        print(f"  Yaw: {ctrl.get_position('yaw'):.4f}°")
        print(f"  Y_coarse: {ctrl.get_position('y_coarse'):.3f} mm")
        print(f"  Y_fine: {ctrl.get_position('y_fine'):.3f} μm")
        print(f"  Z_coarse: {ctrl.get_position('z_coarse'):.3f} μm")
        print(f"  X_fine: {ctrl.get_position('x_fine'):.3f} μm")
        
        # Setup for external trigger scanning
        print("\n--- External trigger setup ---")
        """
        z_dev = ctrl.z_open_setup()
        y_dev = ctrl.y_open_setup()
        
        print("Connect function generators to:")
        print("  - PDXC2 (Z_coarse) analog input")
        print("  - LPS710E/M (Y_fine) analog input")
        
        # Set static voltages if needed
        ctrl.set_voltage('z_open', 37.5)
        ctrl.set_voltage('y_open', 37.5)
        """
        
        print("\n✓ Alignment complete")