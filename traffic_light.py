from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QGridLayout, QHBoxLayout
from PyQt5.QtGui import QPixmap, QColor, QPainter
from PyQt5.QtCore import QTimer, Qt
from opcua import Client, ua
import sys
from collections import deque

# Traffic light widget
# Traffic light widget
class TrafficLightWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.state = "red"  # Initial state
        self.setMinimumSize(60, 100)  # Reduced size for compact layout

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setBrush(Qt.black)
        painter.drawRect(0, 0, self.width(), self.height())

        colors = {"red": QColor("red"), "yellow": QColor("yellow"), "green": QColor("green"), "off": QColor("gray"), "red-yellow": QColor("orange")}

        # Draw the lights
        radius = 20  # Fixed size for bulbs
        offsets = [10, 40, 70]  # Fixed spacing
        states = ["red", "yellow", "green"]

        for i, light_state in enumerate(states):
            color = colors[light_state] if light_state == self.state or (self.state == "red-yellow" and light_state in ["red", "yellow"]) else colors["off"]
            painter.setBrush(color)
            painter.drawEllipse((self.width() - radius) // 2, offsets[i], radius, radius)

    def update_state(self, state):
        self.state = state
        self.update()

class HistoryGraph(QWidget):
    def __init__(self):
        super().__init__()
        self.history = deque(["off"] * 600, maxlen=600)  # Maintain 600 states for 10 seconds of history at 10 Hz
        self.setMinimumSize(1200, 30)  # Set height to approximately 30 pixels

    def add_state(self, state):
        self.history.append(state)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setBrush(Qt.white)
        painter.drawRect(self.rect())  # Draw the background

        colors = {"red": QColor("red"), "yellow": QColor("yellow"), "green": QColor("green"), "off": QColor("gray"), "red-yellow": QColor("orange")}
        bar_width = max(3, self.width() // 600)  # Ensure bar_width is at least 3

        for i, state in enumerate(self.history):
            painter.setBrush(colors.get(state, QColor("gray")))
            painter.drawRect(i * bar_width, 0, bar_width, self.height())

# Main window
class TrafficLightHMI(QMainWindow):
    def __init__(self, num_lights=5, opc_url="opc.tcp://localhost:4840"):
        super().__init__()
        self.setWindowTitle("Traffic Light HMI")
        self.num_lights = num_lights
        self.opc_client = Client(opc_url)
        self.opc_client.set_user("admin")
        self.opc_client.set_password("wago")

        try:
            self.opc_client.connect()
        except Exception as e:
            print(f"Error connecting to OPC UA server: {e}")
            sys.exit(1)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        # Top row for traffic lights
        self.light_row = QHBoxLayout()
        self.layout.addLayout(self.light_row)

        self.traffic_lights = [TrafficLightWidget() for _ in range(self.num_lights)]
        for i, light in enumerate(self.traffic_lights):
            light_container = QVBoxLayout()
            light_container.addWidget(light)
            light_container.addWidget(QLabel(str(i + 1), alignment=Qt.AlignCenter))
            self.light_row.addLayout(light_container)

        # Bottom section for history graphs
        self.graphs = [HistoryGraph() for _ in range(self.num_lights)]
        for i, graph in enumerate(self.graphs):
            graph_container = QHBoxLayout()
            graph_container.addWidget(QLabel(str(i + 1)))
            graph_container.addWidget(graph)
            self.layout.addLayout(graph_container)

        # Define the output nodes for each traffic light
        self.output_nodes = []
        for i in range(self.num_lights+2):
            if i + 1 in [6, 7]:
                continue
            self.output_nodes.append({
                "red": f"ns=4;s=|var|WAGO 750-8211 PFC200 G2 2ETH 2SFP XTR.Application.MainControl.xLight{i + 1}1Red",
                "yellow": f"ns=4;s=|var|WAGO 750-8211 PFC200 G2 2ETH 2SFP XTR.Application.MainControl.xLight{i + 1}1Yellow",
                "green": f"ns=4;s=|var|WAGO 750-8211 PFC200 G2 2ETH 2SFP XTR.Application.MainControl.xLight{i + 1}1Green"
            })

        # Timer to update states
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_states)
        self.timer.start(100)  # Update every 100 ms (10 Hz)

    def update_states(self):
        try:
            for i, light in enumerate(self.traffic_lights):
                red_node = self.opc_client.get_node(self.output_nodes[i]["red"])
                yellow_node = self.opc_client.get_node(self.output_nodes[i]["yellow"])
                green_node = self.opc_client.get_node(self.output_nodes[i]["green"])

                red_state = red_node.get_value()
                yellow_state = yellow_node.get_value()
                green_state = green_node.get_value()

                if red_state and yellow_state:
                    state = "red-yellow"
                elif red_state:
                    state = "red"
                elif yellow_state:
                    state = "yellow"
                elif green_state:
                    state = "green"
                else:
                    state = "off"

                light.update_state(state)
                self.graphs[i].add_state(state)

        except Exception as e:
            print(f"Error during OPC UA state update: {e}")

# Main function
if __name__ == "__main__":
    app = QApplication(sys.argv)

    num_lights = 6  # Adjust the number of lights here
    window = TrafficLightHMI(num_lights=num_lights, opc_url="opc.tcp://192.168.1.101:4840")
    window.show()

    sys.exit(app.exec_())