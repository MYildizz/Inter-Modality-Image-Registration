
import cv2
import os
import numpy as np
import math

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMessageBox

from keras.models import load_model
from skimage import filters
from skimage import feature


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        self.initComponents(MainWindow)
        self.initLogic()

    def initLogic(self):
        #Top buttons
        self.btn_upload_ct.clicked.connect(self._uploadCtBtnClicked)
        self.btn_upload_mr.clicked.connect(self._uploadMrBtnClicked)
        self.btn_make_regisration.clicked.connect(self._makeRegisrationBtnClicked)
        self.btn_clear.clicked.connect(self._clearComponents)

        self.pushButton.clicked.connect(self._homeShowBtnClicked)
        self.btn_ct_show.clicked.connect(self._ctShowBtnClicked)
        self.btn_mr_show.clicked.connect(self._mrShowBtnClicked)
        self.btn_result_show.clicked.connect(self._resultShowBtnClicked)
        self.btn_about_show.clicked.connect(self._aboutShowBtnClicked)

    def _homeShowBtnClicked(self):
        self.stackedWidget.setCurrentWidget(self.page_home)
    
    def _ctShowBtnClicked(self):
        self.stackedWidget.setCurrentWidget(self.page_ct)

    def _mrShowBtnClicked(self):
        self.stackedWidget.setCurrentWidget(self.page_mr)

    def _resultShowBtnClicked(self):
        self.stackedWidget.setCurrentWidget(self.page_result)

    def _aboutShowBtnClicked(self):
        self.stackedWidget.setCurrentWidget(self.page_about)

    def _uploadCtBtnClicked(self):
        
        file_name = QtWidgets.QFileDialog.getOpenFileName(None, 'Open file',
                                            'c:\\', "Image files (*.jpg *.gif, *.jpeg)")

        image_path = file_name[0]        
        ctPixMap = QtGui.QPixmap(image_path)

        if not ctPixMap.isNull():
            self.lbl_ct_title.setVisible(True)
            self.lbl_ct.setPixmap(ctPixMap)
            self.lbl_ct.setVisible(True)

            self._CT = self._uploadImageAsOpenCv(image_path)
        
    def _uploadMrBtnClicked(self):
        file_name = QtWidgets.QFileDialog.getOpenFileName(None, 'Open file',
                                            'c:\\', "Image files (*.jpg *.gif, *.jpeg)")

        image_path = file_name[0]        
        mrPixmap = QtGui.QPixmap(image_path)
        
        if not mrPixmap.isNull():
            self.lbl_mr_title.setVisible(True)
            self.lbl_mr.setPixmap(mrPixmap)
            self.lbl_mr.setVisible(True)

            self._MR = self._uploadImageAsOpenCv(image_path)

    def _makeRegisrationBtnClicked(self):

        best=load_model("best_result.h5")

        test_x=self._CT
        test_x2=self._MR

        #Feature Extraction 1
        feature1_mr = filters.sobel(test_x2).reshape(-1,1)
        feature1_ct=filters.sobel(test_x).reshape(-1,1)
        
        #Feature Extraction 2
        feature2_mr = feature.canny(test_x2).reshape(-1,1)
        feature2_ct=feature.canny(test_x).reshape(-1,1)
        
        #Feature3 Low Pass Filter
        feature3_mr=self.low_pass(test_x2).reshape(-1,1)
        feature3_ct=self.low_pass(test_x).reshape(-1,1)

        test_x = test_x.reshape(-1,1)
        test_x2 = test_x2.reshape(-1,1)
        
        test=np.concatenate((test_x, test_x2,feature1_mr,feature1_ct,feature2_mr,feature2_ct,feature3_mr,feature3_ct), axis=1)
        
        y_pred=best.predict(test)

        size=int(math.sqrt(len(y_pred)))

        y_pred=y_pred.reshape(size,size)

        y_pred=y_pred.astype("uint8")   
        image = y_pred
        image_ = QtGui.QImage(image, image.shape[1],image.shape[0], image.shape[1] ,QtGui.QImage.Format_Grayscale8)
        pix = QtGui.QPixmap(image_)

        self.lbl_result.setPixmap(pix)
        self.lbl_result.setVisible(True)
        self.lbl_result_title.setVisible(True)

        self._show_success_popup('Regisration Succesfull !!')

    
    def _show_success_popup(self, message):
        msg = QMessageBox()
        msg.setText(message)
        msg.setIcon(QMessageBox.Information)

        msg.exec_()

    def _clearComponents(self):
        self.stackedWidget.setCurrentWidget(self.page_home)

        self.lbl_ct_title.setVisible(False)
        self.lbl_ct.setVisible(False)
        self.lbl_mr_title.setVisible(False)
        self.lbl_mr.setVisible(False)
        self.lbl_result_title.setVisible(False)
        self.lbl_result.setVisible(False)



    def initComponents(self,MainWindow):
        self.path = os.path.dirname(os.path.abspath(__file__))


        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1021, 749)
        MainWindow.setMinimumSize(QtCore.QSize(800, 550))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setStyleSheet("background:rgb(91,90,90);")
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.frame_top = QtWidgets.QFrame(self.centralwidget)
        self.frame_top.setMaximumSize(QtCore.QSize(16777215, 55))
        self.frame_top.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_top.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame_top.setObjectName("frame_top")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.frame_top)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.frame_toodle = QtWidgets.QFrame(self.frame_top)
        self.frame_toodle.setMinimumSize(QtCore.QSize(90, 55))
        self.frame_toodle.setMaximumSize(QtCore.QSize(90, 55))
        self.frame_toodle.setStyleSheet("background:rgb(0,143,150);")
        self.frame_toodle.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_toodle.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame_toodle.setObjectName("frame_toodle")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.frame_toodle)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setSpacing(0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.toodle = QtWidgets.QPushButton(self.frame_toodle)
        self.toodle.setMinimumSize(QtCore.QSize(95, 55))
        self.toodle.setMaximumSize(QtCore.QSize(95, 55))
        self.toodle.setStyleSheet("")
        self.toodle.setText("")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(os.path.join(self.path,"icons/aybuIcon.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.toodle.setIcon(icon)
        self.toodle.setIconSize(QtCore.QSize(50, 50))
        self.toodle.setFlat(True)
        self.toodle.setObjectName("toodle")
        self.horizontalLayout_3.addWidget(self.toodle)
        self.horizontalLayout.addWidget(self.frame_toodle)
        self.frame_top_east = QtWidgets.QFrame(self.frame_top)
        self.frame_top_east.setMinimumSize(QtCore.QSize(930, 55))
        self.frame_top_east.setMaximumSize(QtCore.QSize(930, 55))
        self.frame_top_east.setStyleSheet("background:rgb(51,51,51);")
        self.frame_top_east.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_top_east.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame_top_east.setObjectName("frame_top_east")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.frame_top_east)
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_4.setSpacing(0)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.frame_top_east1 = QtWidgets.QFrame(self.frame_top_east)
        self.frame_top_east1.setMinimumSize(QtCore.QSize(90, 55))
        self.frame_top_east1.setMaximumSize(QtCore.QSize(90, 55))
        self.frame_top_east1.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_top_east1.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_top_east1.setObjectName("frame_top_east1")
        self.btn_upload_ct = QtWidgets.QPushButton(self.frame_top_east1)
        self.btn_upload_ct.setGeometry(QtCore.QRect(4, 10, 84, 31))
        self.btn_upload_ct.setMinimumSize(QtCore.QSize(84, 0))
        self.btn_upload_ct.setMaximumSize(QtCore.QSize(84, 16777215))
        font = QtGui.QFont()
        font.setPointSize(7)
        self.btn_upload_ct.setFont(font)
        self.btn_upload_ct.setStyleSheet("QPushButton {\n    border: 2px solid rgb(51,51,51);\n    border-radius: 5px;    \n    color:rgb(255,255,255);\n    background-color: rgb(84,84,84);\n}\nQPushButton:hover {\n    border: 2px solid rgb(102,178,255);\n    background-color: rgb(102,178,255);\n}\nQPushButton:pressed {    \n    border: 2px solid rgb(0,143,150);\n    background-color: rgb(51,51,51);\n}\n\nQPushButton:disabled {    \n    border-radius: 5px;    \n    border: 2px solid rgb(112,112,112);\n    background-color: rgb(112,112,112);\n}")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(os.path.join(self.path,"icons/max.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btn_upload_ct.setIcon(icon1)
        self.btn_upload_ct.setIconSize(QtCore.QSize(16, 16))
        self.btn_upload_ct.setObjectName("btn_upload_ct")
        self.horizontalLayout_4.addWidget(self.frame_top_east1)
        self.frame_top_east2 = QtWidgets.QFrame(self.frame_top_east)
        self.frame_top_east2.setMinimumSize(QtCore.QSize(90, 55))
        self.frame_top_east2.setMaximumSize(QtCore.QSize(90, 55))
        self.frame_top_east2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_top_east2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_top_east2.setObjectName("frame_top_east2")
        self.btn_upload_mr = QtWidgets.QPushButton(self.frame_top_east2)
        self.btn_upload_mr.setGeometry(QtCore.QRect(0, 10, 84, 31))
        self.btn_upload_mr.setMinimumSize(QtCore.QSize(84, 0))
        self.btn_upload_mr.setMaximumSize(QtCore.QSize(84, 16777215))
        font = QtGui.QFont()
        font.setPointSize(7)
        self.btn_upload_mr.setFont(font)
        self.btn_upload_mr.setStyleSheet("QPushButton {\n"
                "    border: 2px solid rgb(51,51,51);\n"
                "    border-radius: 5px;    \n"
                "    color:rgb(255,255,255);\n"
                "    background-color: rgb(84,84,84);\n"
                "}\n"
                "QPushButton:hover {\n"
                "    border: 2px solid rgb(0,143,150);\n"
                "    background-color: rgb(0,143,150);\n"
                "}\n"
                "QPushButton:pressed {    \n"
                "    border: 2px solid rgb(0,143,150);\n"
                "    background-color: rgb(51,51,51);\n"
                "}\n"
                "\n"
                "QPushButton:disabled {    \n"
                "    border-radius: 5px;    \n"
                "    border: 2px solid rgb(112,112,112);\n"
                "    background-color: rgb(112,112,112);\n"
                "}")
        self.btn_upload_mr.setIcon(icon1)
        self.btn_upload_mr.setObjectName("btn_upload_mr")
        self.horizontalLayout_4.addWidget(self.frame_top_east2)
        self.frame_top_east3 = QtWidgets.QFrame(self.frame_top_east)
        self.frame_top_east3.setMinimumSize(QtCore.QSize(95, 55))
        self.frame_top_east3.setMaximumSize(QtCore.QSize(95, 55))
        self.frame_top_east3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_top_east3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_top_east3.setObjectName("frame_top_east3")
        self.btn_make_regisration = QtWidgets.QPushButton(self.frame_top_east3)
        self.btn_make_regisration.setGeometry(QtCore.QRect(0, 10, 95, 31))
        self.btn_make_regisration.setMinimumSize(QtCore.QSize(95, 0))
        self.btn_make_regisration.setMaximumSize(QtCore.QSize(95, 16777215))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.btn_make_regisration.setFont(font)
        self.btn_make_regisration.setStyleSheet("QPushButton {\n"
        "    border: 2px solid rgb(51,51,51);\n"
        "    border-radius: 5px;    \n"
        "    color:rgb(255,255,255);\n"
        "    background-color: rgb(84,84,84);\n"
        "}\n"
        "QPushButton:hover {\n"
        "    border: 2px solid rgb(0,153,76);\n"
        "    background-color: rgb(0,153,76);\n"
        "}\n"
        "QPushButton:pressed {    \n"
        "    border: 2px solid rgb(0,143,150);\n"
        "    background-color: rgb(51,51,51);\n"
        "}\n"
        "\n"
        "QPushButton:disabled {    \n"
        "    border-radius: 5px;    \n"
        "    border: 2px solid rgb(112,112,112);\n"
        "    background-color: rgb(112,112,112);\n"
        "}")
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(os.path.join(self.path,"icons/settAsset 50.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btn_make_regisration.setIcon(icon2)
        self.btn_make_regisration.setObjectName("btn_make_regisration")
        self.horizontalLayout_4.addWidget(self.frame_top_east3)
        self.frame_top_east4 = QtWidgets.QFrame(self.frame_top_east)
        self.frame_top_east4.setMinimumSize(QtCore.QSize(90, 55))
        self.frame_top_east4.setMaximumSize(QtCore.QSize(90, 55))
        self.frame_top_east4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_top_east4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_top_east4.setObjectName("frame_top_east4")
        self.btn_clear = QtWidgets.QPushButton(self.frame_top_east4)
        self.btn_clear.setGeometry(QtCore.QRect(0, 10, 84, 31))
        self.btn_clear.setMinimumSize(QtCore.QSize(84, 0))
        self.btn_clear.setMaximumSize(QtCore.QSize(84, 16777215))
        font = QtGui.QFont()
        font.setPointSize(7)
        self.btn_clear.setFont(font)
        self.btn_clear.setStyleSheet("QPushButton {\n    border: 2px solid rgb(51,51,51);\n    border-radius: 5px;    \n    color:rgb(255,255,255);\n    background-color: rgb(84,84,84);\n}\nQPushButton:hover {\n    border: 2px solid rgb(204,0,0);\n    background-color: rgb(204,0,0);\n}\nQPushButton:pressed {    \n    border: 2px solid rgb(0,143,150);\n    background-color: rgb(51,51,51);\n}\n\nQPushButton:disabled {    \n    border-radius: 5px;    \n    border: 2px solid rgb(112,112,112);\n    background-color: rgb(112,112,112);\n}")
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(os.path.join(self.path,"icons/cleanAsset 59.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btn_clear.setIcon(icon3)
        self.btn_clear.setObjectName("btn_clear")
        self.horizontalLayout_4.addWidget(self.frame_top_east4)
        self.horizontalLayout.addWidget(self.frame_top_east)
        self.verticalLayout.addWidget(self.frame_top)
        self.frame_bottom = QtWidgets.QFrame(self.centralwidget)
        self.frame_bottom.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_bottom.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame_bottom.setObjectName("frame_bottom")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.frame_bottom)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setSpacing(0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.frame_bottom_west = QtWidgets.QFrame(self.frame_bottom)
        self.frame_bottom_west.setMinimumSize(QtCore.QSize(90, 0))
        self.frame_bottom_west.setMaximumSize(QtCore.QSize(90, 16777215))
        self.frame_bottom_west.setStyleSheet("background:rgb(51,51,51);")
        self.frame_bottom_west.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_bottom_west.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame_bottom_west.setObjectName("frame_bottom_west")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.frame_bottom_west)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setSpacing(0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.frame_home = QtWidgets.QFrame(self.frame_bottom_west)
        self.frame_home.setMinimumSize(QtCore.QSize(80, 70))
        self.frame_home.setMaximumSize(QtCore.QSize(160, 70))
        self.frame_home.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_home.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame_home.setObjectName("frame_home")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.frame_home)
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_4.setSpacing(0)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.pushButton = QtWidgets.QPushButton(self.frame_home)
        self.pushButton.setMinimumSize(QtCore.QSize(80, 70))
        self.pushButton.setMaximumSize(QtCore.QSize(160, 70))
        font = QtGui.QFont()
        font.setFamily("Segoe UI Black")
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton.setFont(font)
        self.pushButton.setStyleSheet("QPushButton {\n    border: none;\n    background-color: rgba(0,0,0,0);\n    color: white;\n}\nQPushButton:hover {\n    background-color: rgb(91,90,90);\n}\nQPushButton:pressed {    \n    background-color: rgba(0,0,0,0);\n}")
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap(os.path.join(self.path,"icons/homeAsset 46.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton.setIcon(icon4)
        self.pushButton.setIconSize(QtCore.QSize(45, 45))
        self.pushButton.setObjectName("pushButton")
        self.verticalLayout_4.addWidget(self.pushButton)
        self.verticalLayout_3.addWidget(self.frame_home)
        self.fram_ct_show = QtWidgets.QFrame(self.frame_bottom_west)
        self.fram_ct_show.setMinimumSize(QtCore.QSize(80, 70))
        self.fram_ct_show.setMaximumSize(QtCore.QSize(165, 70))
        self.fram_ct_show.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.fram_ct_show.setFrameShadow(QtWidgets.QFrame.Plain)
        self.fram_ct_show.setObjectName("fram_ct_show")
        self.horizontalLayout_15 = QtWidgets.QHBoxLayout(self.fram_ct_show)
        self.horizontalLayout_15.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_15.setSpacing(0)
        self.horizontalLayout_15.setObjectName("horizontalLayout_15")
        self.btn_ct_show = QtWidgets.QPushButton(self.fram_ct_show)
        self.btn_ct_show.setMinimumSize(QtCore.QSize(80, 70))
        self.btn_ct_show.setMaximumSize(QtCore.QSize(160, 70))
        font = QtGui.QFont()
        font.setFamily("Segoe UI Black")
        font.setPointSize(7)
        font.setBold(True)
        font.setWeight(75)
        self.btn_ct_show.setFont(font)
        self.btn_ct_show.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.btn_ct_show.setAutoFillBackground(False)
        self.btn_ct_show.setStyleSheet("QPushButton {\n    border: none;\n    background-color: rgba(0,0,0,0);\n    color: white;\n}\nQPushButton:hover {\n    background-color: rgb(91,90,90);\n}\nQPushButton:pressed {    \n    background-color: rgba(0,0,0,0);\n}")
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap(os.path.join(self.path,"icons/ctIcon.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btn_ct_show.setIcon(icon5)
        self.btn_ct_show.setIconSize(QtCore.QSize(40, 40))
        self.btn_ct_show.setFlat(True)
        self.btn_ct_show.setObjectName("btn_ct_show")
        self.horizontalLayout_15.addWidget(self.btn_ct_show)
        self.verticalLayout_3.addWidget(self.fram_ct_show)
        self.frame_mr_show = QtWidgets.QFrame(self.frame_bottom_west)
        self.frame_mr_show.setMinimumSize(QtCore.QSize(80, 70))
        self.frame_mr_show.setMaximumSize(QtCore.QSize(160, 70))
        self.frame_mr_show.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_mr_show.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame_mr_show.setObjectName("frame_mr_show")
        self.horizontalLayout_16 = QtWidgets.QHBoxLayout(self.frame_mr_show)
        self.horizontalLayout_16.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_16.setSpacing(0)
        self.horizontalLayout_16.setObjectName("horizontalLayout_16")
        self.btn_mr_show = QtWidgets.QPushButton(self.frame_mr_show)
        self.btn_mr_show.setMinimumSize(QtCore.QSize(80, 70))
        self.btn_mr_show.setMaximumSize(QtCore.QSize(160, 70))
        font = QtGui.QFont()
        font.setFamily("Segoe UI Black")
        font.setPointSize(7)
        font.setBold(True)
        font.setWeight(75)
        self.btn_mr_show.setFont(font)
        self.btn_mr_show.setStyleSheet("QPushButton {\nborder: none;\nbackground-color: rgba(0,0,0,0);\ncolor: white;\n}\nQPushButton:hover {\nbackground-color: rgb(91,90,90);\n}\nQPushButton:pressed {\nbackground-color: rgba(0,0,0,0);\n}")
        icon6 = QtGui.QIcon()
        icon6.addPixmap(QtGui.QPixmap(os.path.join(self.path,"icons/mrIcon.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btn_mr_show.setIcon(icon6)
        self.btn_mr_show.setIconSize(QtCore.QSize(40, 40))
        self.btn_mr_show.setFlat(True)
        self.btn_mr_show.setObjectName("btn_mr_show")
        self.horizontalLayout_16.addWidget(self.btn_mr_show)
        self.verticalLayout_3.addWidget(self.frame_mr_show)
        self.frame_result_show = QtWidgets.QFrame(self.frame_bottom_west)
        self.frame_result_show.setMinimumSize(QtCore.QSize(80, 70))
        self.frame_result_show.setMaximumSize(QtCore.QSize(160, 70))
        self.frame_result_show.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_result_show.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame_result_show.setObjectName("frame_result_show")
        self.horizontalLayout_17 = QtWidgets.QHBoxLayout(self.frame_result_show)
        self.horizontalLayout_17.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_17.setSpacing(0)
        self.horizontalLayout_17.setObjectName("horizontalLayout_17")
        self.btn_result_show = QtWidgets.QPushButton(self.frame_result_show)
        self.btn_result_show.setMinimumSize(QtCore.QSize(80, 70))
        self.btn_result_show.setMaximumSize(QtCore.QSize(160, 70))
        font = QtGui.QFont()
        font.setFamily("Segoe UI Black")
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        self.btn_result_show.setFont(font)
        self.btn_result_show.setStyleSheet("QPushButton {\nborder: none;\n    background-color: rgba(0,0,0,0);\n    color: white;\n}\nQPushButton:hover {\n    background-color: rgb(91,90,90);\n}\nQPushButton:pressed {    \n    background-color: rgba(0,0,0,0);\n}")
        icon7 = QtGui.QIcon()
        icon7.addPixmap(QtGui.QPixmap(os.path.join(self.path,"icons/resultIcon.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btn_result_show.setIcon(icon7)
        self.btn_result_show.setIconSize(QtCore.QSize(45, 45))
        self.btn_result_show.setFlat(True)
        self.btn_result_show.setObjectName("btn_result_show")
        self.horizontalLayout_17.addWidget(self.btn_result_show)
        self.verticalLayout_3.addWidget(self.frame_result_show)
        self.frame_about_us = QtWidgets.QFrame(self.frame_bottom_west)
        self.frame_about_us.setMinimumSize(QtCore.QSize(80, 65))
        self.frame_about_us.setMaximumSize(QtCore.QSize(160, 70))
        self.frame_about_us.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_about_us.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame_about_us.setObjectName("frame_about_us")
        self.horizontalLayout_18 = QtWidgets.QHBoxLayout(self.frame_about_us)
        self.horizontalLayout_18.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_18.setSpacing(0)
        self.horizontalLayout_18.setObjectName("horizontalLayout_18")
        self.btn_about_show = QtWidgets.QPushButton(self.frame_about_us)
        self.btn_about_show.setMinimumSize(QtCore.QSize(80, 70))
        self.btn_about_show.setMaximumSize(QtCore.QSize(160, 70))
        font = QtGui.QFont()
        font.setFamily("Segoe UI Black")
        font.setPointSize(7)
        font.setBold(True)
        font.setWeight(75)
        self.btn_about_show.setFont(font)
        self.btn_about_show.setStyleSheet("QPushButton {\n    border: none;\n    background-color: rgba(0,0,0,0);\n    color: white;\n}\nQPushButton:hover {\n    background-color: rgb(91,90,90);\n}\nQPushButton:pressed {    \n    background-color: rgba(0,0,0,0);\n}")
        icon8 = QtGui.QIcon()
        icon8.addPixmap(QtGui.QPixmap(os.path.join(self.path,"icons/aboutIconv2.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btn_about_show.setIcon(icon8)
        self.btn_about_show.setIconSize(QtCore.QSize(38, 38))
        self.btn_about_show.setFlat(True)
        self.btn_about_show.setObjectName("btn_about_show")
        self.horizontalLayout_18.addWidget(self.btn_about_show)
        self.verticalLayout_3.addWidget(self.frame_about_us)
        self.horizontalLayout_2.addWidget(self.frame_bottom_west)
        self.frame_bottom_east = QtWidgets.QFrame(self.frame_bottom)
        self.frame_bottom_east.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_bottom_east.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame_bottom_east.setObjectName("frame_bottom_east")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.frame_bottom_east)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setSpacing(0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.frame = QtWidgets.QFrame(self.frame_bottom_east)
        self.frame.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame.setObjectName("frame")
        self.horizontalLayout_14 = QtWidgets.QHBoxLayout(self.frame)
        self.horizontalLayout_14.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_14.setSpacing(0)
        self.horizontalLayout_14.setObjectName("horizontalLayout_14")
        self.stackedWidget = QtWidgets.QStackedWidget(self.frame)
        self.stackedWidget.setObjectName("stackedWidget")
        self.page_home = QtWidgets.QWidget()
        self.page_home.setObjectName("page_home")
        self.stackedWidget.addWidget(self.page_home)
        self.page_ct = QtWidgets.QWidget()
        self.page_ct.setObjectName("page_ct")
        self.lbl_ct_title = QtWidgets.QLabel(self.page_ct)
        self.lbl_ct_title.setGeometry(QtCore.QRect(290, 10, 281, 16))
        font = QtGui.QFont()
        font.setFamily("Perpetua")
        font.setPointSize(14)
        font.setBold(False)
        font.setWeight(50)
        font.setStyleStrategy(QtGui.QFont.PreferAntialias)
        self.lbl_ct_title.setFont(font)
        self.lbl_ct_title.setStyleSheet("QLabel{\n    color : white;\n}")
        self.lbl_ct_title.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_ct_title.setWordWrap(False)
        self.lbl_ct_title.setObjectName("lbl_ct_title")
        self.lbl_ct = QtWidgets.QLabel(self.page_ct)
        self.lbl_ct.setGeometry(QtCore.QRect(100, 40, 721, 581))
        self.lbl_ct.setText("")
        self.lbl_ct.setScaledContents(True)
        self.lbl_ct.setObjectName("lbl_ct")
        self.stackedWidget.addWidget(self.page_ct)
        self.page_mr = QtWidgets.QWidget()
        self.page_mr.setObjectName("page_mr")
        self.lbl_mr_title = QtWidgets.QLabel(self.page_mr)
        self.lbl_mr_title.setGeometry(QtCore.QRect(290, 10, 281, 16))
        font = QtGui.QFont()
        font.setFamily("Perpetua")
        font.setPointSize(14)
        self.lbl_mr_title.setFont(font)
        self.lbl_mr_title.setStyleSheet("QLabel{color : white;}")
        self.lbl_mr_title.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_mr_title.setObjectName("lbl_mr_title")
        self.lbl_mr = QtWidgets.QLabel(self.page_mr)
        self.lbl_mr.setGeometry(QtCore.QRect(100, 40, 721, 581))
        self.lbl_mr.setText("")
        self.lbl_mr.setObjectName("lbl_mr")
        self.lbl_mr.setScaledContents(True)
        self.stackedWidget.addWidget(self.page_mr)
        self.page_result = QtWidgets.QWidget()
        self.page_result.setObjectName("page_result")
        self.lbl_result_title = QtWidgets.QLabel(self.page_result)
        self.lbl_result_title.setGeometry(QtCore.QRect(290, 10, 281, 16))
        font = QtGui.QFont()
        font.setFamily("Perpetua")
        font.setPointSize(14)
        self.lbl_result_title.setFont(font)
        self.lbl_result_title.setStyleSheet("QLabel{color : white;}")
        self.lbl_result_title.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_result_title.setObjectName("lbl_result_title")
        self.lbl_result = QtWidgets.QLabel(self.page_result)
        self.lbl_result.setGeometry(QtCore.QRect(100, 40, 721, 581))
        self.lbl_result.setText("")
        self.lbl_result.setObjectName("lbl_result")
        self.stackedWidget.addWidget(self.page_result)
        self.page_about = QtWidgets.QWidget()
        self.page_about.setObjectName("page_about")
        self.lbl_about = QtWidgets.QLabel(self.page_about)
        self.lbl_about.setGeometry(QtCore.QRect(60, 270, 811, 61))
        font = QtGui.QFont()
        font.setFamily("Segoe UI Black")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.lbl_about.setFont(font)
        self.lbl_about.setStyleSheet("QLabel{color : white;}")
        self.lbl_about.setTextFormat(QtCore.Qt.PlainText)
        self.lbl_about.setScaledContents(True)
        self.lbl_about.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_about.setObjectName("lbl_about")
        self.lbl_ybu_icon = QtWidgets.QLabel(self.page_about)
        self.lbl_ybu_icon.setGeometry(QtCore.QRect(320, 40, 261, 221))
        self.lbl_ybu_icon.setText("")
        self.lbl_ybu_icon.setPixmap(QtGui.QPixmap(os.path.join(self.path,"icons/aybuIcon.png")))
        self.lbl_ybu_icon.setScaledContents(True)
        self.lbl_ybu_icon.setObjectName("lbl_ybu_icon")
        self.lbl_authors = QtWidgets.QLabel(self.page_about)
        self.lbl_authors.setGeometry(QtCore.QRect(50, 360, 281, 191))
        font = QtGui.QFont()
        font.setFamily("Segoe UI Black")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.lbl_authors.setFont(font)
        self.lbl_authors.setStyleSheet("QLabel{color : white;}")
        self.lbl_authors.setTextFormat(QtCore.Qt.PlainText)
        self.lbl_authors.setScaledContents(True)
        self.lbl_authors.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.lbl_authors.setObjectName("lbl_authors")
        self.stackedWidget.addWidget(self.page_about)
        self.horizontalLayout_14.addWidget(self.stackedWidget)
        self.verticalLayout_2.addWidget(self.frame)
        self.frame_low = QtWidgets.QFrame(self.frame_bottom_east)
        self.frame_low.setMinimumSize(QtCore.QSize(0, 20))
        self.frame_low.setMaximumSize(QtCore.QSize(16777215, 20))
        self.frame_low.setStyleSheet("")
        self.frame_low.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_low.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame_low.setObjectName("frame_low")
        self.horizontalLayout_11 = QtWidgets.QHBoxLayout(self.frame_low)
        self.horizontalLayout_11.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_11.setSpacing(0)
        self.horizontalLayout_11.setObjectName("horizontalLayout_11")
        self.frame_tab = QtWidgets.QFrame(self.frame_low)
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        self.frame_tab.setFont(font)
        self.frame_tab.setStyleSheet("background:rgb(51,51,51);")
        self.frame_tab.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_tab.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame_tab.setObjectName("frame_tab")
        self.horizontalLayout_12 = QtWidgets.QHBoxLayout(self.frame_tab)
        self.horizontalLayout_12.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_12.setSpacing(0)
        self.horizontalLayout_12.setObjectName("horizontalLayout_12")
        self.lab_tab = QtWidgets.QLabel(self.frame_tab)
        font = QtGui.QFont()
        font.setFamily("Segoe UI Light")
        font.setPointSize(10)
        self.lab_tab.setFont(font)
        self.lab_tab.setStyleSheet("color:rgb(255,255,255);")
        self.lab_tab.setObjectName("lab_tab")
        self.horizontalLayout_12.addWidget(self.lab_tab)
        self.horizontalLayout_11.addWidget(self.frame_tab)
        self.frame_drag = QtWidgets.QFrame(self.frame_low)
        self.frame_drag.setMinimumSize(QtCore.QSize(20, 20))
        self.frame_drag.setMaximumSize(QtCore.QSize(20, 20))
        self.frame_drag.setStyleSheet("background:rgb(51,51,51);")
        self.frame_drag.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_drag.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame_drag.setObjectName("frame_drag")
        self.horizontalLayout_13 = QtWidgets.QHBoxLayout(self.frame_drag)
        self.horizontalLayout_13.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_13.setSpacing(0)
        self.horizontalLayout_13.setObjectName("horizontalLayout_13")
        self.horizontalLayout_11.addWidget(self.frame_drag)
        self.verticalLayout_2.addWidget(self.frame_low)
        self.horizontalLayout_2.addWidget(self.frame_bottom_east)
        self.verticalLayout.addWidget(self.frame_bottom)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        self.stackedWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self._clearComponents()
        self._CT = None
        self._MR = None
        


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Image Registrator"))
        self.btn_upload_ct.setText(_translate("MainWindow", "Ct Image"))
        self.btn_upload_mr.setText(_translate("MainWindow", "Mr Image"))
        self.btn_make_regisration.setText(_translate("MainWindow", "Regisration"))
        self.btn_clear.setText(_translate("MainWindow", "Clear!"))
        self.pushButton.setText(_translate("MainWindow", "Home"))
        self.btn_ct_show.setToolTip(_translate("MainWindow", "Home"))
        self.btn_ct_show.setText(_translate("MainWindow", "Ct Image"))
        self.btn_mr_show.setToolTip(_translate("MainWindow", "Bug"))
        self.btn_mr_show.setText(_translate("MainWindow", "Mr Image"))
        self.btn_result_show.setToolTip(_translate("MainWindow", "Cloud"))
        self.btn_result_show.setText(_translate("MainWindow", "Result"))
        self.btn_about_show.setToolTip(_translate("MainWindow", "Android"))
        self.btn_about_show.setText(_translate("MainWindow", "About Us"))
        self.lbl_ct_title.setText(_translate("MainWindow", "CT IMAGE"))
        self.lbl_mr_title.setText(_translate("MainWindow", "MR IMAGE"))
        self.lbl_result_title.setText(_translate("MainWindow", "RESULT IMAGE"))
        self.lbl_about.setText(_translate("MainWindow", "This application is a product of Ankara Yıldırım Beyazıt University\n2020-2021 Academic Year Computer Engineering graduation project."))
        self.lbl_authors.setText(_translate("MainWindow", "+Gurkan ALTINTAŞ (Developer)\n\n+Oğuzhan TOKLU (Developer)\n\n+Berkan YILDIRIM (Yatıyo Amk K*rdu)\n\n+Muratcan YILDIZ (Project Manager)\n\n+Baha ŞEN (Project Supervisor)"))
        self.lab_tab.setText(_translate("MainWindow", "<html><head/><body><p><br/></p></body></html>"))
        self.frame_drag.setToolTip(_translate("MainWindow", "Drag"))

    def _uploadImageAsOpenCv(self, image_path):
        _image = cv2.imread(image_path)
        image = cv2.cvtColor(_image, cv2.COLOR_BGR2GRAY)
        return image
    
    def low_pass(self,img):       
        original = np.fft.fft2(img)
        center = np.fft.fftshift(original)
        LowPassCenter =  center * self.calculateIdealLowPassFilter(30,img.shape)
    
        LowPass = np.fft.ifftshift(LowPassCenter)
        inverse_LowPass = np.fft.ifft2(LowPass)   
    
        return np.abs(inverse_LowPass)



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
