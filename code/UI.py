# -*- coding: UTF-8 -*-
from tkinter import *
import matplotlib
matplotlib.use("TkAgg")
from PIL import Image, ImageTk
from geturl import *
from type import *
import time

#URL：www.solarpeng.com/
#www.solarpeng.com/?cat=<script>alert(1)<script>



class MY_GUI():

    def __init__(self,parent_init_name):
        self.parent_init_name = parent_init_name
        self.ListIP = []
        self.ListInformation = []
        self.new_url_list = []

    def set_init_windows(self):
        self.parent_init_name.title("web入侵检测系统")
        self.parent_init_name.geometry('700x700+10+10')
        self.parent_init_name.resizable(width=True, height=True)

        self.ImageFrame = Frame(self.parent_init_name)
        self.ImgFrame = LabelFrame(self.ImageFrame, text="Warning!!!")
        self.im = Image.open("../image/warning.jpeg")
        self.img = ImageTk.PhotoImage(self.im)
        Label(self.ImgFrame, width=627, height=220, image=self.img).grid(row=0, column=0, columnspan=3)
        self.ImgFrame.pack()
        self.ImageFrame.pack()

        self.textframe = Frame(self.parent_init_name)
        self. developer = LabelFrame(self.textframe, text="开发信息")
        self.developer.pack(padx=10, pady=10)
        self.textframe.pack()
        Label(self.developer, width=70, height=2, bg="white", text="开发人员：郭文博\n").grid(column=0, row=0)
        Label(self.developer, width=70, height=2, bg="white", text="开发环境：Mac os\n").grid(column=0, row=1)
        Label(self.developer, width=70, height=2, bg="white", text="开发时间：2018-1-05\n").grid(column=0, row=2)
        Label(self.developer, width=70, height=2, bg="white", text="版本号：D-1.00\n").grid(column=0, row=3)


    #  扫描的局域网主机ip信息
        self.IpFrame = Frame(self.parent_init_name)
        self.iplistframe = LabelFrame(self.IpFrame,text = "实时入侵检测结果显示")
        self.iplistframe.pack(padx=10, pady=10)
        self.listip = Listbox(self.iplistframe,width=70,bd = 0)
        # self.listip.bind('<Double-Button-1>', self.output_information)
        for item in self.ListIP:
            self.listip.insert(END,item)
        self.listip.pack()
        self.IpFrame.pack()

    #  UI界面按钮
        self.buttonframe = Frame(self.parent_init_name)
        Button(self.buttonframe,text="开始", command=self.printhello).grid(column=0, row=0)
        Button(self.buttonframe, text="退出", command=self.exitui).grid(column=1, row=0)
        Button(self.buttonframe, text="帮助", command=self.help).grid(column=2, row=0)
        self.buttonframe.pack()

    def help(self):

        top = Toplevel()
        top.title("帮助")
        top.geometry('600x600')

        ImageFrame = Frame(top)
        ImgFrame = LabelFrame(ImageFrame,text = "help")
        im = Image.open("../image/1.jpg")
        img = ImageTk.PhotoImage(im)
        Label(ImgFrame, width=350,height=160,image=img).grid(row=0, column=0,columnspan=3)
        ImgFrame.pack()
        ImageFrame.pack()

        # pilImage = Image.open("1.png")
        # tkImage = ImageTk.PhotoImage(image=pilImage)
        # label = Label(top,image=tkImage)
        # label.pack()

        textframe = Frame(top)
        developer = LabelFrame(textframe, text="使用教程")
        developer.pack(padx=10, pady=10)
        textframe.pack()
        Label(developer, width=50, height=2, bg="white", text="环境配置：python3.6\n").grid(column=0, row=0)
        Label(developer, width=50, height=2, bg="white", text="开始按钮：开始检测URL\n").grid(column=0, row=1)
        Label(developer, width=50, height=2, bg="white", text="退出按钮：点击即可退出系统程序                      \n").grid(column=0,row=2)
        Label(developer, width=50, height=2, bg="white", text="帮助按钮：提示程序如何进行使用                      \n").grid(column=0,row=3)
        Label(developer, width=50, height=2, bg="white", text="检测情况：该区域会显示系统检测的url分类结果，\n").grid(column=0, row=4)

        top.mainloop()

    def output_information(self):

        pass
    def get_url(self,event):

        pass

    def printhello(self):

        while True:

            Sniffer()
            if (test != []):
                self.new_url_list = test[0]
                url_list = []
                new_url = unquote(test[0][4][4:]).lower()
                print(new_url)
                new_url_split = split_word(new_url)
                print(new_url_split)
                url_list.append(new_url)
                print('____')
                print(url_list)
                a = Train()
                url_type = find_type(new_url_split)
                result = a.predict(url_list)
                self.new_url_list.append(result)
                self.new_url_list.append(url_type)
                print(self.new_url_list)
                time.sleep(3)
                
            self.add_information()


    def add_information(self):

        self.listip.delete(0, END)
        for i in self.new_url_list:
            self.listip.insert(END, i)

    def exitui(self):

        exit(0)


def UI_start():

    init_windows = Tk()
    ZMJ_PORTAL = MY_GUI(init_windows)
    ZMJ_PORTAL.set_init_windows()
    init_windows.mainloop()
