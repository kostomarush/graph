import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QGraphicsScene, QGraphicsView, QGraphicsEllipseItem, QGraphicsLineItem, QVBoxLayout, QWidget, QPushButton, QComboBox, QAction, QHBoxLayout, 
                            QGraphicsTextItem, QInputDialog, QDialog,QMessageBox, QTableWidget, QTableWidgetItem, QLabel, QFileDialog, QGridLayout, QStatusBar)
from PyQt5.QtCore import Qt, QPointF, QLineF, QRectF
from PyQt5.QtGui import QPen, QColor
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
import csv
import re
import heapq
import copy

class GraphApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Сеть")
        self.setGeometry(100, 100, 850, 650)
        self.create_menu()
        self.create_scene()
        self.create_layout()
        self.all_items = []
        self.connect_items_graph = []
        self.selected_items = []
        self.index = 0
        self.minm = 0

    def create_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu('Файл')
        function_menu = menubar.addMenu('Функции')
        open_file = QAction('Открыть файл', self)
        open_file.setShortcut('Ctrl+U')
        open_file.triggered.connect(self.load_csv_file)
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
        matrix_graf = QAction('Интегральные показатели', self)
        clear_derev = QAction('Очистить поле', self)
        clear_derev.triggered.connect(self.clear_der)
        function_menu.addAction(ost_derev)
        ost_derev.triggered.connect(self.show_ostovnoe_derevo_page)
        function_menu.addAction(center_derev)
        center_derev.triggered.connect(self.show_center)
        function_menu.addAction(matrix_graf)
        function_menu.addAction(clear_derev)
        matrix_graf.triggered.connect(self.show_integ_pokaz)

    def show_ostovnoe_derevo_page(self):
        if len(self.connect_items_graph) == 0:
            QMessageBox.critical(self, "Ошибка", "Добавьте элементы графа")
            return
        dialog = OstovnoeDerevoPage(self.connect_items_graph, self)
        dialog.exec_()

    def show_integ_pokaz(self):
        if len(self.connect_items_graph) == 0:
            QMessageBox.critical(self, 'Ошибка', 'Добавьте элементы графа')
            return
        dialog = IntegrPokaz(self.connect_items_graph, self)
        dialog.exec_()

    def show_center(self):
        if len(self.connect_items_graph) == 0:
            QMessageBox.critical(self, "Ошибка", "Добавьте элементы графа")
            return
        calculator = ShortestPathMatrixCalculator(self.connect_items_graph)
        adjacency_matrix = calculator.compute_adjacency_matrix()
        d0_matrix = calculator.compute_d0_matrix(adjacency_matrix)
        shortest_path_matrix = calculator.compute_shortest_path_matrix(adjacency_matrix)
        center = calculator.compute_center(shortest_path_matrix)    
        window = MatrixWindow(d0_matrix, shortest_path_matrix, center)
        window.exec_()

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

    def load_csv_file(self):
        pointer_set = set()
        csv_file, _ = QFileDialog.getOpenFileName(self, "Выберите файл с графом", "", "CSV Files (*.csv)")
        if csv_file:
            with open(csv_file, newline='') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    pointer_1 = int(row['pointer_1'])
                    xy_1_str = row['xy_1']
                    numbers = re.findall(r'[-+]?\d*\.\d+|\d+', xy_1_str)
                    xy_1 = QPointF(float(numbers[1]), float(numbers[2]))
                    pointer_2 = int(row['pointer_2'])
                    xy_2_str = row['xy_2']
                    numbers2 = re.findall(r'[-+]?\d*\.\d+|\d+', xy_2_str)
                    xy_2 = QPointF(float(numbers2[1]), float(numbers2[2]))
                    weight = int(row['weight'])
                    connect_type = row['connect_type']
                    item1 = QGraphicsEllipseItem(xy_1.x() - 10, xy_1.y() - 10, 30, 30)
                    item1.setBrush(QColor(Qt.cyan))
                    self.scene.addItem(item1)
                    index1 = self.scene.items().index(item1)
                    text_item1 = QGraphicsTextItem(str(pointer_1))
                    text_item1.setPos(xy_1.x() - 5, xy_1.y() - 5)
                    self.scene.addItem(text_item1)
                    item2 = QGraphicsEllipseItem(xy_2.x() - 10, xy_2.y() - 10, 30, 30)
                    item2.setBrush(QColor(Qt.cyan))
                    self.scene.addItem(item2)
                    index2 = self.scene.items().index(item2)
                    text_item2 = QGraphicsTextItem(str(pointer_2))
                    text_item2.setPos(xy_2.x() - 5, xy_2.y() - 5)
                    self.scene.addItem(text_item2)
                    line_pen = QPen(Qt.black)
                    line = self.scene.addLine(xy_1.x(), xy_1.y(), xy_2.x(), xy_2.y(), line_pen)
                    text = QGraphicsTextItem(f"{weight} {connect_type}")
                    text.setPos((xy_1 + xy_2) / 2)
                    self.scene.addItem(text)
                    self.connect_items_graph.append({
                    'pointer_1': pointer_1,
                    'xy_1' : xy_1,
                    'pointer_2': pointer_2,
                    'xy_2' : xy_2,
                    'weight': weight,
                    'connect_type' : connect_type
                    })
                    pointer_set.add(pointer_1)
                    pointer_set.add(pointer_2)
                    if item1 not in self.all_items:
                        self.all_items.append(item1)
                for pointer in pointer_set:
                    self.combobox1.addItem(str(pointer))
                    self.combobox2.addItem(str(pointer))


    def write_dict_to_csv(self, data, filename):
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['pointer_1', 'xy_1', 'pointer_2', 'xy_2', 'weight', 'connect_type']
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
                    'xy_1' : p1,
                    'pointer_2': index2,
                    'xy_2' : p2,
                    'weight': weight,
                    'connect_type' : connection_type
                    })
                    self.scene.addItem(line)
                    self.scene.addItem(text)

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

class IntegrPokaz(QDialog):
    def __init__(self, graph, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Интегральный показатель качества ОД")
        self.graph = graph
        self.layout = QVBoxLayout()
        self.table1 = QTableWidget()
        self.layout.addWidget(self.table1)
        self.setLayout(self.layout)
        znach = self.find_all_spanning_trees(self.graph)

        integral_quality = self.calculate_integral_quality(znach)
        self.table1.setRowCount(len(integral_quality))
        self.table1.setColumnCount(len(integral_quality[0]))
        self.table1.resizeColumnsToContents()
        column_labels = ["kLi", "kni", "khi", "sum"]
        self.table1.setHorizontalHeaderLabels(column_labels)
        for i in range(len(integral_quality)):
            for j in range(len(integral_quality[0])):
                item = QTableWidgetItem(str(integral_quality[i][j]))
                item.setFlags(Qt.ItemIsEnabled)
                self.table1.setItem(i, j, item)

    def find_all_spanning_trees(self, graph):
        vertices = set()
        for edge in graph:
            vertices.add(edge['pointer_1'])
            vertices.add(edge['pointer_2'])
        starting_vertex = vertices.pop()
        visited = {starting_vertex}
        heap = [(0, None, starting_vertex)]
        while heap:
            weight, parent, current_vertex = heapq.heappop(heap)
            if parent is not None:
                yield {'pointer_1': parent, 'pointer_2': current_vertex, 'weight': weight}
            for edge in graph:
                next_edge = None
                if edge['pointer_1'] == current_vertex:
                    next_edge = edge
                elif edge['pointer_2'] == current_vertex:
                    next_edge = edge
                if next_edge is not None:
                    next_vertex = next_edge['pointer_1'] if next_edge['pointer_1'] != current_vertex else next_edge['pointer_2']
                    if next_vertex not in visited:
                        visited.add(next_vertex)
                        heapq.heappush(heap, (next_edge['weight'], current_vertex, next_vertex))
        return vertices

    def compute_quality_parameters(self,graph):
        
        # Сначала определим список всех уникальных узлов в графе
        all_nodes = set()
        for edge in graph:
            all_nodes.add(edge['pointer_1'])
            all_nodes.add(edge['pointer_2'])

        # Инициализируем словарь для хранения свойств каждого остовного дерева
        spanning_trees_properties = []

        # Для каждого узла в графе будем строить остовное дерево
        for node in all_nodes:
            # Инициализируем переменные для хранения общего веса и количества узлов
            total_weight = 0
            total_nodes = 1  # Начинаем с 1, так как текущий узел уже включен в остовное дерево

            # Создаем список, в который будем добавлять узлы остовного дерева
            spanning_tree_nodes = [node]

            # Перебираем все рёбра в графе
            for edge in graph:
                # Если ребро соединяет текущий узел с другим узлом остовного дерева
                if edge['pointer_1'] in spanning_tree_nodes and edge['pointer_2'] not in spanning_tree_nodes:
                    # Добавляем вес ребра к общему весу
                    total_weight += edge['weight']
                    # Добавляем новый узел к остовному дереву
                    spanning_tree_nodes.append(edge['pointer_2'])
                    # Увеличиваем счетчик узлов
                    total_nodes += 1
                elif edge['pointer_2'] in spanning_tree_nodes and edge['pointer_1'] not in spanning_tree_nodes:
                    # Добавляем вес ребра к общему весу
                    total_weight += edge['weight']
                    # Добавляем новый узел к остовному дереву
                    spanning_tree_nodes.append(edge['pointer_1'])
                    # Увеличиваем счетчик узлов
                    total_nodes += 1

            # Добавляем свойства остовного дерева в словарь
            spanning_trees_properties.append((total_weight, total_nodes, total_weight*total_nodes))

        return spanning_trees_properties

    def calculate_integral_quality(self, min_spanning_trees):
        quality_parameters = self.compute_quality_parameters(min_spanning_trees)
        quality_parameters.sort(key=lambda x: (x[0], x[1], x[2]))
        L_max, L_min = quality_parameters[-1][0], quality_parameters[0][0]
        n_max, n_min = quality_parameters[-1][1], quality_parameters[0][1]
        h_max, h_min = quality_parameters[-1][2], quality_parameters[0][2]
        quality_indices = []
        for L, n, h in quality_parameters:
            try:
                kLi = (L_max - L) / (L_max - L_min)
            except:
                kLi = 0
            try:
                kni = (n_max - n) / (n_max - n_min)
            except:
                kni = 0
            try:
                khi = (h_max - h) / (h_max - h_min)
            except:
                khi = 0
            itog_znach = kLi + kni + khi
            quality_indices.append((round(kLi,2), round(kni,2), round(khi,2), round(itog_znach,2)))
        return quality_indices

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

class MatrixWindow(QDialog):
    def __init__(self, matrix1, matrix2, center):
        super().__init__()
        self.setWindowTitle("Матрицы смежности")
        self.grid_layout = QGridLayout(self)

        self.table1 = QTableWidget()
        self.table2 = QTableWidget()
        self.statusbar = QStatusBar()

        self.grid_layout.addWidget(QLabel("D[0]"), 0, 0)
        self.grid_layout.addWidget(self.table1, 1, 0)

        self.grid_layout.addWidget(QLabel("Матрица кратчайших расстояний"), 0, 1)
        self.grid_layout.addWidget(self.table2, 1, 1)
        
        self.grid_layout.addWidget(self.statusbar, 2, 0, 1, 2)

        self.populate_table(self.table1, matrix1)
        self.populate_table(self.table2, matrix2)
        
        self.statusbar.showMessage(f"\nЦентр графа находится в вершине {center}")

    def populate_table(self, table, matrix):
        table.setRowCount(len(matrix))
        table.setColumnCount(len(matrix[0]))
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                item = QTableWidgetItem(str(matrix[i][j]))
                item.setFlags(Qt.ItemIsEnabled)
                table.setItem(i, j, item)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GraphApp()
    window.show()
    sys.exit(app.exec_())