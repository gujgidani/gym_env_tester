from sumolib.translation import SUMO_HOME
from sumolib import checkBinary
import traci
import os, sys
from statistics import mean
from pymodbus.server import StartTcpServer
from pymodbus.datastore import ModbusSequentialDataBlock, ModbusSlaveContext, ModbusServerContext
from pymodbus.client import ModbusTcpClient
import threading
from time import sleep
from opcua import Client, ua


start_signal = False
signal_status = [False] * 18
modbus_client = None
sim_step = 0


class ModbusServer:
    def __init__(self, host="192.168.1.161", port=502):
        self.host = host
        self.port = port
        self.store = ModbusSlaveContext(
            di=ModbusSequentialDataBlock(0, [0]*100),
            co=ModbusSequentialDataBlock(0, [0]*100),
            hr=ModbusSequentialDataBlock(0, [0]*100),
            ir=ModbusSequentialDataBlock(0, [0]*100))
        self.context = ModbusServerContext(slaves=self.store, single=True)

    def start(self):
        self.server_thread = threading.Thread(target=self._run_server)
        self.server_thread.daemon = True
        self.server_thread.start()

    def _run_server(self):
        StartTcpServer(context=self.context, address=(self.host, self.port))

    def update_values(self, occupancy, vehicle_count):
        # Convert floating point occupancy values to integers (multiply by 100 to preserve 2 decimal places)
        occupancy_int = [int(x * 100) for x in occupancy]
        # Update holding registers with occupancy values
        self.store.setValues(3, 1, occupancy_int)
        # Update holding registers with vehicle count values
        self.store.setValues(3, len(occupancy) + 1, vehicle_count)
        # Signal update completion
        self.store.setValues(3, 0, [1])

class SubHandler(object):
    def datachange_notification(self, node, val, data):
        node_id = node.nodeid  # vagy node.browse_name.name a név alapú ellenőrzéshez
        #print(f"Node ID: {node_id}, Value: {val}")
        if node_id.Identifier == "|var|WAGO 750-8211 PFC200 G2 2ETH 2SFP XTR.Application.MainControl.oSecEnding":
            if val == True:
                #modbud read input rengisters 0-1
                registers = modbus_client.read_input_registers(0, count = 2)

                signal_status = modbus_client.convert_from_registers(registers.registers, modbus_client.DATATYPE.BITS)[0:18]
                for i in range(int(len(signal_status) / 8 + 0.99)):
                    print(i)
                    print(signal_status[i * 8:(i + 1) * 8])
                temp = signal_status[0:8]
                signal_status[0:8] = signal_status[8:16]
                signal_status[8:16] = temp
                #print(signal_status)

                traci.simulationStep()
                global sim_step
                sim_step += 1
                print(f"Step: {sim_step}")
        if node_id.Identifier == "|var|WAGO 750-8211 PFC200 G2 2ETH 2SFP XTR.Application.GVL_Case_1.eState":
            if val == 3:
                global start_signal
                start_signal = True
                print("Start signal received")



def get_traffic_info(traffic_flow):
    occupancy = []
    vehicle_count = []

    for idx in range(len(traffic_flow["occupancy"])):
        try:
            occupancy.append(mean(traffic_flow["occupancy"][idx]))
        except Exception:
            occupancy.append(0)

        try:
            vehicle_count.append(sum(traffic_flow["vehicle_count"][idx]))
        except Exception:
            vehicle_count.append(0)

    return occupancy, vehicle_count

if __name__ == "__main__":
    if 'SUMO_HOME' in os.environ:
        sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))

    # Initialize Modbus server
    modbus_server = ModbusServer()
    modbus_server.start()
    print("Modbus server started on localhost:502")

    render_mode = "gui"

    if render_mode == "console":
        if sys.platform == "darwin":
            sumo_binary = "/opt/homebrew/Cellar/sumo/1.20.0/bin/sumo"
        elif sys.platform == "win32":
            sumo_binary = "C:\\Program Files (x86)\\Eclipse\\Sumo\\bin\\sumo.exe"
        else:
            sumo_binary = '/usr/bin/sumo'
    else:
        if sys.platform == "darwin":
            sumo_binary = "/opt/homebrew/Cellar/sumo/1.20.0/bin/sumo-gui"
        elif sys.platform == "win32":
            sumo_binary = "C:\\Program Files (x86)\\Eclipse\\Sumo\\bin\\sumo-gui.exe"
        else:
            sumo_binary = '/usr/bin/sumo-gui'

    simulation_time = 3600
    detectors = []
    current_step = 0
    edge_list = ("660942467#1","-24203041#0","660942464")

    sumo_cmd = [
        sumo_binary,
        "-c",
        "sumo_files/osm.sumocfg",
        "--start",
        "-e", str(simulation_time),
        "--quit-on-end",
    ]

    traci.start(sumo_cmd)
    detectors = traci.inductionloop.getIDList()
    # Initialize traffic flow data structure
    traffic_flow = {
        "vehicle_count": [[] for _ in range(len(detectors))],
        "occupancy": [[] for _ in range(len(detectors))]
    }



    step = 0
    opc_client = Client("opc.tcp://192.168.1.101:4840")
    opc_client.set_user("admin")
    opc_client.set_password("wago")
    opc_client.connect()

    modbus_client = ModbusTcpClient("192.168.1.101")
    modbus_client.connect()


    #while not start_signal:
    #    sleep(0.1)

    try:
        handler = SubHandler()
        subscription = opc_client.create_subscription(50, handler)
        nodes = [
            opc_client.get_node("ns=4;s=|var|WAGO 750-8211 PFC200 G2 2ETH 2SFP XTR.Application.MainControl.oSecEnding"),
            opc_client.get_node("ns=4;s=|var|WAGO 750-8211 PFC200 G2 2ETH 2SFP XTR.Application.GVL_Case_1.eState"),
        ]
        handles = [subscription.subscribe_data_change(node) for node in nodes]

        for handle in handles:
            subscription.modify_monitored_item(handle, 25)
        print("Subscription created")

        #while True:
        #    sleep(0.1)

        while sim_step < 3600:
            if start_signal:
                if sim_step % 5 == 4:
                    # Add traffic measurements to traffic_flow
                    for idx, detector in enumerate(detectors):
                        traffic_flow["vehicle_count"][idx].append(traci.inductionloop.getLastStepVehicleNumber(detector))
                        traffic_flow["occupancy"][idx].append(traci.inductionloop.getLastStepOccupancy(detector))
                if sim_step % 300 == 295:
                #if sim_step % 100 == 95:
                    # Get traffic information
                    occupancy, vehicle_count = get_traffic_info(traffic_flow)

                    # Update Modbus server values
                    modbus_server.update_values(occupancy, vehicle_count)


    except Exception as e:
        print(f"Error in main loop: {e}")
    finally:
        # Cleanup
        try:
            if 'subscription' in locals():
                subscription.delete()
            opc_client.disconnect()
        except Exception as e:
            print(f"Error during cleanup: {e}")

        traci.close()
        print("Simulation ended")
