import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QGraphicsScene, QGraphicsView, QGraphicsEllipseItem, QGraphicsLineItem, QVBoxLayout, QWidget, QPushButton, QComboBox, QAction, QHBoxLayout, 
                            QGraphicsTextItem, QInputDialog, QDialog,QMessageBox, QTableWidget, QTableWidgetItem, QLabel, QFileDialog)
from PyQt5.QtCore import Qt, QPointF, QLineF
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
import csv

class GraphApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interactive Graph Application")
        self.setGeometry(100, 100, 850, 650)
        self.create_menu()
        self.create_scene()
        self.create_layout()
        self.all_items = []
        self.connect_items_graph = []
        self.selected_items = []

    def create_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu('Файл')
        function_menu = menubar.addMenu('Функции')
        open_file = QAction('Открыть файл', self)
        open_file.setShortcut('Ctrl+U')
        save_file = QAction('Сохранить файл', self)
        save_file.setShortcut('Ctrl+S')
        save_file.triggered.connect(self.save_csv_dialog)
        exit_action = QAction('Закрыть', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(open_file)
        file_menu.addAction(save_file)
        file_menu.addAction(exit_action)
        ost_derev =QAction('Построение остовного дерева', self)
        center_derev = QAction('Определение центра', self)
        clear_derev = QAction('Очистить поле', self)
        clear_derev.triggered.connect(self.clear_der)
        function_menu.addAction(ost_derev)
        ost_derev.triggered.connect(self.show_ostovnoe_derevo_page)
        function_menu.addAction(center_derev)
        center_derev.triggered.connect(self.show_center)
        function_menu.addAction(clear_derev)

    def show_ostovnoe_derevo_page(self):
        if len(self.connect_items_graph) == 0:
            QMessageBox.critical(self, "Ошибка", "Добавьте элементы графа")
            return
        dialog = OstovnoeDerevoPage(self.connect_items_graph, self)
        dialog.exec_()

    def show_center(self):
        if len(self.connect_items_graph) == 0:
            QMessageBox.critical(self, "Ошибка", "Добавьте элементы графа")
            return
        calculator = ShortestPathMatrixCalculator(self.connect_items_graph)
        adjacency_matrix = calculator.compute_adjacency_matrix()
        print("Матрица смежности:")
        for row in adjacency_matrix:
            print(row)
        
        d0_matrix = calculator.compute_d0_matrix(adjacency_matrix)
        print("\nМатрица d[0]:")
        for row in d0_matrix:
            print(row)

        shortest_path_matrix = calculator.compute_shortest_path_matrix(adjacency_matrix)
        print("\nМатрица длин кратчайших путей:")
        for row in shortest_path_matrix:
            print(row)

        center = calculator.compute_center(shortest_path_matrix)
        print(f"\nЦентр графа находится в вершине {center}")

    def create_scene(self):
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.scene.setSceneRect(0, 0, 800, 600)
        self.view.mousePressEvent = self.mousePressEvent

    def create_layout(self):
        self.widget = QWidget()
        self.layout = QHBoxLayout(self.widget)
        node_layout = QVBoxLayout()
        self.connect_button = QPushButton("Соединить")
        self.connect_button.clicked.connect(self.connect_items)
        node_layout.addWidget(self.connect_button)
        self.combobox1 = QComboBox()
        node_layout.addWidget(self.combobox1)
        self.combobox2 = QComboBox()
        node_layout.addWidget(self.combobox2)
        self.layout.addLayout(node_layout)
        self.layout.addWidget(self.view)
        self.setCentralWidget(self.widget)

    def mousePressEvent(self, event):
        pos = self.view.mapToScene(event.pos())
        if event.button() == Qt.LeftButton:
            item = QGraphicsEllipseItem(pos.x() - 10, pos.y() - 10, 30, 30)
            item.setBrush(Qt.cyan)
            self.scene.addItem(item)
            self.all_items.append(item)
            index = len(self.all_items) - 1
            self.combobox1.addItem(str(index))
            self.combobox2.addItem(str(index))
            text_item = QGraphicsTextItem(str(index))
            text_item.setPos(pos.x() - 5, pos.y() - 5)  
            self.scene.addItem(text_item)

    def write_dict_to_csv(self, data, filename):
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['pointer_1', 'pointer_2', 'weight', 'connect_type']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in data:
                writer.writerow(row)

    def save_csv_dialog(self, data):
        if len(self.connect_items_graph) == 0:
            QMessageBox.critical(self, "Ошибка", "Постройте граф")
            return
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getSaveFileName(None, "Сохранить схему", "", "CSV Files (*.csv)", options=options)
        if filename:
            self.write_dict_to_csv(self.connect_items_graph, filename)


    def connect_items(self):
        index1 = self.combobox1.currentIndex()
        index2 = self.combobox2.currentIndex()
        if index1 != -1 and index2 != -1 and index1 != index2:
            item1 = self.all_items[index1]
            item2 = self.all_items[index2]
            weight, ok = QInputDialog.getInt(self, "", "Растояние между узлами:", 0, 0, 100, 1)
            if ok:
                connection_types = ['Eth', 'СЦИ']
                connection_type, ok = QInputDialog.getItem(self, "", "Тип соединения:", connection_types, 0, False)
                if ok:
                    p1 = item1.rect().center()
                    p2 = item2.rect().center()
                    line = QGraphicsLineItem(QLineF(p1, p2))
                    line.weight = weight
                    text = QGraphicsTextItem(f"{weight} {connection_type}")
                    text.setPos((p1 + p2) / 2)
                    self.connect_items_graph.append({
                    'pointer_1': index1,
                    'pointer_2': index2,
                    'weight': weight,
                    'connect_type' : connection_type
                    })
                    self.scene.addItem(line)
                    self.scene.addItem(text)
                    print(self.connect_items_graph)



    def clear_der(self):
        self.scene.clear()
        self.all_items.clear()  
        self.connect_items_graph.clear()
        self.combobox1.clear()  
        self.combobox2.clear()

    def closeEvent(self, event):
        self.widget.deleteLater()

class ShortestPathMatrixCalculator:
    def __init__(self, graph):
        self.graph = graph

    def compute_adjacency_matrix(self):
        num_vertices = max(max(edge['pointer_1'], edge['pointer_2']) for edge in self.graph) + 1
        adjacency_matrix = [[float('inf')] * num_vertices for _ in range(num_vertices)]
        for edge in self.graph:
            pointer_1, pointer_2, weight = edge['pointer_1'], edge['pointer_2'], edge['weight']
            adjacency_matrix[pointer_1][pointer_2] = weight
            adjacency_matrix[pointer_2][pointer_1] = weight
        for i in range(num_vertices):
            adjacency_matrix[i][i] = 0
        return adjacency_matrix

    def compute_d0_matrix(self, adjacency_matrix):
        num_vertices = len(adjacency_matrix)
        d0_matrix = [[0] * num_vertices for _ in range(num_vertices)]

        for i in range(num_vertices):
            for j in range(num_vertices):
                if i != j and adjacency_matrix[i][j] == float('inf'):
                    d0_matrix[i][j] = float('inf')
                else:
                    d0_matrix[i][j] = adjacency_matrix[i][j]

        return d0_matrix

    def compute_shortest_path_matrix(self, adjacency_matrix):
        num_vertices = len(adjacency_matrix)
        shortest_path_matrix = [[0] * num_vertices for _ in range(num_vertices)]

        for i in range(num_vertices):
            for j in range(num_vertices):
                shortest_path_matrix[i][j] = adjacency_matrix[i][j]

        for k in range(num_vertices):
            for i in range(num_vertices):
                for j in range(num_vertices):
                    if shortest_path_matrix[i][k] != float('inf') and shortest_path_matrix[k][j] != float('inf') \
                            and shortest_path_matrix[i][k] + shortest_path_matrix[k][j] < shortest_path_matrix[i][j]:
                        shortest_path_matrix[i][j] = shortest_path_matrix[i][k] + shortest_path_matrix[k][j]

        return shortest_path_matrix

    def compute_center(self, shortest_path_matrix):
        num_vertices = len(shortest_path_matrix)
        min_sum = float('inf')
        center = -1

        for i in range(num_vertices):
            sum_distances = sum(shortest_path_matrix[i])
            if sum_distances < min_sum:
                min_sum = sum_distances
                center = i

        return center






class OstovnoeDerevoPage(QDialog):
    def __init__(self, graph, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Остовные деревья")
        self.graph = graph
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.calculate_and_display_ostovnoe_derevo()

    def calculate_and_display_ostovnoe_derevo(self):
        edges = self.graph
        num_vertices = max(max(edge['pointer_1'], edge['pointer_2']) for edge in edges) + 1
        adjacency_matrix = [[0] * num_vertices for _ in range(num_vertices)]
        for edge in edges:
            pointer_1, pointer_2, weight = edge['pointer_1'], edge['pointer_2'], edge['weight']
            adjacency_matrix[pointer_1][pointer_2] = weight
            adjacency_matrix[pointer_2][pointer_1] = weight
        self.display_adjacency_matrix(adjacency_matrix)
        G = nx.Graph()
        for i in range(num_vertices):
            for j in range(i + 1, num_vertices):
                if adjacency_matrix[i][j] != 0:
                    G.add_edge(i, j)
        
        mst_edges = nx.minimum_spanning_edges(G)
        num_ostovnoe_derevo = len(list(mst_edges))    
        self.layout.addWidget(QLabel(f"Количество остовных деревьев: {num_ostovnoe_derevo}"))
        mst_edges1 = list(nx.minimum_spanning_edges(G))
        mst = nx.Graph()
        mst.add_edges_from(mst_edges1)
        plt.figure()
        pos = nx.spring_layout(mst)
        nx.draw(mst, pos, with_labels=True, node_color='skyblue', node_size=500, font_size=12, font_weight='bold')
        plt.title("Остовное дерево")
        plt.show()


    def display_adjacency_matrix(self, adjacency_matrix):
        self.table_widget = QTableWidget()
        self.table_widget.setRowCount(len(adjacency_matrix))
        self.table_widget.setColumnCount(len(adjacency_matrix[0]))
        for i in range(len(adjacency_matrix)):
            for j in range(len(adjacency_matrix[0])):
                if adjacency_matrix[i][j] != 0:
                    item = QTableWidgetItem(str('1'))
                else:
                    item = QTableWidgetItem(str('0'))
                self.table_widget.setItem(i, j, item)
        self.layout.addWidget(self.table_widget)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GraphApp()
    window.show()
    sys.exit(app.exec_())
