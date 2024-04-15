import sys
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import (QApplication, QMainWindow, QGraphicsScene, QGraphicsView, QGraphicsEllipseItem, QGraphicsLineItem, QVBoxLayout, QWidget, QPushButton, QComboBox, QAction, QHBoxLayout, 
                            QGraphicsTextItem, QInputDialog, QDialog,QMessageBox, QTableWidget, QTableWidgetItem, QLabel, QFileDialog, QGridLayout, QStatusBar, QSpacerItem, QSizePolicy)
from PyQt5.QtCore import Qt, QPointF, QLineF, QRectF, QThread, pyqtSignal
from PyQt5.QtGui import QPen, QColor
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
import csv
import re
import os
import scipy as sp


ico = os.path.join(sys._MEIPASS, "icon.ico") if getattr(sys, 'frozen', False) else "icon.ico"
app1 = QtWidgets.QApplication(sys.argv)
app1.setWindowIcon(QtGui.QIcon(ico))

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
        spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        node_layout.addItem(spacer)
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
                connection_types = ['Eth', 'СЦИ', 'ПЦИ']
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
        znach, G = self.count_spanning_trees(self.graph)
        total_L, total_n, total_H = self.calculate_integral_quality(znach, G)
        self.table1.setRowCount(len(total_L))
        self.table1.setColumnCount(4)
        self.table1.resizeColumnsToContents()
        column_labels = ["kLi", "kni", "khi", "sum"]
        self.table1.setHorizontalHeaderLabels(column_labels)
        for i in range(len(total_L)):
            self.table1.setItem(i, 0, QTableWidgetItem(str(round(total_L[i],2))))
            self.table1.setItem(i, 1, QTableWidgetItem(str(round(total_n[i],2))))
            self.table1.setItem(i, 2, QTableWidgetItem(str(round(total_H[i],2))))
            self.table1.setItem(i, 3, QTableWidgetItem(str(round((total_H[i] + total_L[i] + total_n[i]), 2))))
    
    def count_nodes(self, G, root):
        _sum = 0
        v = set()
        v.add(root)
        k = 2  
        visited = set()  
        while v:
            n_v = set()  
            for _v in v:
                if _v not in visited: 
                    visited.add(_v)  
                    neighbors = G.neighbors(_v)  
                    for neighbor in neighbors:
                        if neighbor not in visited:
                            n_v.add(neighbor)
                            _sum += k
            k += 1
            v = n_v
        return _sum

    def _expand(self, G, explored_nodes, explored_edges):
        frontier_nodes = list()
        frontier_edges = list()
        for v in explored_nodes:
            for u in nx.neighbors(G,v):
                if not (u in explored_nodes):
                    frontier_nodes.append(u)
                    frontier_edges.append([(u,v), (v,u)])
        return zip([explored_nodes | frozenset([v]) for v in frontier_nodes], [explored_edges | frozenset(e) for e in frontier_edges])
        
    def find_all_spanning_trees(self, G, root=0):
        explored_nodes = frozenset([root])
        explored_edges = frozenset([])
        solutions = [(explored_nodes, explored_edges)]
        for ii in range(G.number_of_nodes()-1):
            solutions = [self._expand(G, nodes, edges) for (nodes, edges) in solutions]
            solutions = set([item for sublist in solutions for item in sublist])
        return [nx.from_edgelist(edges) for (nodes, edges) in solutions]

    def count_spanning_trees(self, graph_edges):
        G = nx.Graph()
        for edge in graph_edges:
            pointer_1 = edge['pointer_1']
            pointer_2 = edge['pointer_2']
            weight = edge['weight']
            G.add_edge(pointer_1, pointer_2, weight=weight)
        couner_spaning_trees = self.find_all_spanning_trees(G)
        return couner_spaning_trees, G

    def spanning_treeses(self, graph_edges):
        G = nx.Graph()
        for edge in graph_edges:
            pointer_1 = edge['pointer_1']
            pointer_2 = edge['pointer_2']
            if edge['connect_type'] == 'Eth':
                weight = 2
            elif edge['connect_type'] == 'СЦИ':
                weight = 1
            elif edge['connect_type'] == 'ПЦИ':
                weight = 1.5
            G.add_edge(pointer_1, pointer_2, weight=weight)
        return G

    def calculate_integral_quality(self, spanning_trees, G):
        quality_indices = []
        quality = []
        itog_L = []
        itog_n = []
        itog_h = []
        total_sum = []
        center_node = nx.center(G)[0]
        G1 = self.spanning_treeses(self.graph)
        for tree in spanning_trees:  # Узел, с которого начинается обход в глубину
            total_sum.append(self.count_nodes(tree, center_node))
            weights = [G[edge[0]][edge[1]]['weight'] for edge in tree.edges()]
            n_weights = [G1[edge[0]][edge[1]]['weight'] for edge in tree.edges()]
            total_weight = sum(weights)
            sum_total = sum(n_weights)
            quality_indices.append(total_weight)
            quality.append(sum_total)
        max_total_weight = max(quality_indices)
        min_total_weight = min(quality_indices)
        max_total_sum = max(total_sum)
        min_total_sum = min(total_sum)
        max_total_h = max(quality)
        min_total_h = min(quality)
        for i in quality_indices:
            try:
                itog_L.append((max_total_weight - i) / (max_total_weight - min_total_weight))
            except:
                itog_L.append(0)
        for i in total_sum:
            try:
                itog_n.append((max_total_sum - i) / (max_total_sum - min_total_sum))
            except:
                itog_n.append(0)
        for i in quality:
            try:
                itog_h.append((max_total_h - i) / (max_total_h - min_total_h))
            except:
                itog_h.append(0)
        return itog_L, itog_n, itog_h


class OstovnoeDerevoPage(QDialog):
    def __init__(self, graph, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Остовные деревья")
        self.graph = graph
        self.layout = QVBoxLayout()
        self.statusbar = QStatusBar()
        self.layout.addWidget(self.statusbar)
        self.setLayout(self.layout)
        self.calculate_and_display_ostovnoe_derevo()

    def _expand(self, G, explored_nodes, explored_edges):
        frontier_nodes = list()
        frontier_edges = list()
        for v in explored_nodes:
            for u in nx.neighbors(G,v):
                if not (u in explored_nodes):
                    frontier_nodes.append(u)
                    frontier_edges.append([(u,v), (v,u)])
        return zip([explored_nodes | frozenset([v]) for v in frontier_nodes], [explored_edges | frozenset(e) for e in frontier_edges])
        
    def find_all_spanning_trees(self, G, root=0):
        explored_nodes = frozenset([root])
        explored_edges = frozenset([])
        solutions = [(explored_nodes, explored_edges)]
        for ii in range(G.number_of_nodes()-1):
            solutions = [self._expand(G, nodes, edges) for (nodes, edges) in solutions]
            solutions = set([item for sublist in solutions for item in sublist])
        return [nx.from_edgelist(edges) for (nodes, edges) in solutions]

    def count_spanning_trees(self, graph_edges):
        G = nx.Graph()
        for edge in graph_edges:
            pointer_1 = edge['pointer_1']
            pointer_2 = edge['pointer_2']
            weight = edge['weight']
            G.add_edge(pointer_1, pointer_2, weight=weight)
        couner_spaning_trees = self.find_all_spanning_trees(G)
        return couner_spaning_trees

    def calculate_and_display_ostovnoe_derevo(self):
        edges = self.graph
        num_vertices = max(max(edge['pointer_1'], edge['pointer_2']) for edge in edges) + 1
        adjacency_matrix = [[0] * num_vertices for _ in range(num_vertices)]
        for edge in edges:
            pointer_1, pointer_2, weight = edge['pointer_1'], edge['pointer_2'], edge['weight']
            adjacency_matrix[pointer_1][pointer_2] = weight
            adjacency_matrix[pointer_2][pointer_1] = weight
        self.display_adjacency_matrix(adjacency_matrix)
        total_spanning_trees = self.count_spanning_trees(edges)
        count_spanning_trees = len(total_spanning_trees)
        self.statusbar.showMessage(f"Количество остовных деревьев: {count_spanning_trees}")
        num_rows = count_spanning_trees // 2 + count_spanning_trees % 2
        num_cols = 2 if count_spanning_trees > 1 else 1
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 5))
        for idx, mst_edges in enumerate(total_spanning_trees):
            row = idx // num_cols
            col = idx % num_cols
            mst = nx.Graph()
            mst.add_edges_from(mst_edges.edges)
            pos = nx.spring_layout(mst)
            nx.draw(mst, pos, ax=axes[row, col], with_labels=True, node_color='skyblue', node_size=500, font_size=12, font_weight='bold')
            axes[row, col].set_title(f"Остовное дерево {idx+1}")
        for idx in range(len(total_spanning_trees), num_rows * num_cols):
            row = idx // num_cols
            col = idx % num_cols
            fig.delaxes(axes[row, col])
        plt.tight_layout()
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
        
        self.statusbar.showMessage(f"Центр графа находится в вершине {center}")

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