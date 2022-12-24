import tkinter
import tkinter.messagebox
import customtkinter

customtkinter.set_appearance_mode("System")


class App(customtkinter.CTk):
    WIDTH = 780
    HEIGHT = 520

    def __init__(self):
        super().__init__()
        self.title("Русский сентимент анализ")
        self.geometry(f"{App.WIDTH}x{App.HEIGHT}")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    # def construct_window(self):
    #     pass
    #   тут должна быть магия переопределения

    def on_closing(self, event=0):
        self.destroy()

    def start(self):
        self.mainloop()


class MainWindow(App):
    def __init__(self, parent):
        super().__init__(parent)
        self.self.label_info_1 = customtkinter.CTkLabel(master=self.frame_info,
                                                   text="CTkLabel: Lorem ipsum dolor sit,\n" +
                                                        "amet consetetur sadipscing elitr,\n" +
                                                        "sed diam nonumy eirmod tempor",
                                                   height=100,
                                                   fg_color=("white", "gray38"),  # <- custom tuple-color
                                                   justify=tkinter.LEFT)

    def on_closing(self, event=0):
        pass

    def start(self):
        pass

    """
    Создание первого окна приложения
    """

    def construct_window(self):
        pass


if __name__ == '__main__':
    app = App()
    app.start()
