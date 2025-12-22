# Import the Myo library
import myo
 
 
# Create Listener class that inherits from myo.DeviceListener
class Listener(myo.DeviceListener):
    # Method triggers on when the Myo is paired
    def on_paired(self, event):
        # Print message with the device name
        print("Hello, {}!".format(event.device_name))
        # Trigger short vibration on the bracelet
        event.device.vibrate(myo.VibrationType.short)
 
    # Method triggers when Myo is not connected
    def on_unpaired(self, event):
        print("No paired device present")
        return False  # Stop the hub
 
 
if __name__ == '__main__':
    # Initialize the Myo library, by specifying the path to the SDK
    myo.init(sdk_path='C:/Users/Usuario/Downloads/Proyecto_MYO/Research_Deep_Learning_Rock_Scissor_Paper/MYO_armband_SDK/myo-sdk-win-0.9.0')
    # Create a hub to manage Myo devices
    hub = myo.Hub()
    # Create instance of the Listener class
    listener = Listener()
    # Start an infinite loop to run the hub with the specified listener
    while hub.run(listener.on_event, 500):
        pass