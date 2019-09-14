#
# Copyright (c) 2017, Manfred Constapel
# This file is licensed under the terms of the MIT license.
#

import lib.utl as utl_
import lib.env as env_

import random
import math
import time

import tkinter as tk
import tkinter.ttk as ttk
import tkinter.messagebox as tk_mb
import tkinter.filedialog as tk_fd
import tkinter.font as tk_font

# ----------------------------

events = utl_.Events()  # global event system

# ----------------------------

def init():
    root = tk.Tk()
    general_font = tk_font.nametofont('TkDefaultFont')
    general_font.config(size=8, weight='normal')
    root.option_add('*Font', general_font)
    f = tk_font.Font(name='TkDefaultFont', exists=True, root=root)
    try:
        ttk.Style().theme_use('clam')
    except tk.TclError:
        pass
    ttk.Style().configure('.', font=f)
    return root

# ----------------------------

class BaseFrame(ttk.Frame):

    __widget_cfg = {'relief': tk.GROOVE, 'borderwidth': 2}

    def __init__(self, master, *args, **kwargs):
        ttk.Frame.__init__(self, master, *args, **BaseFrame.__widget_cfg, **kwargs)


class ListFrame(BaseFrame):

    def __init__(self, master, items=None, bg_color=None, padding=5, *args, **kwargs):

        BaseFrame.__init__(self, master, padding=padding)

        relief = self['relief']
        if bg_color == ttk.Style().lookup('TFrame', 'background'):
            relief = tk.FLAT

        box = tk.Listbox(self, borderwidth=self['borderwidth'], relief=relief,
                         highlightthickness=0, selectmode=tk.SINGLE, exportselection=False, *args, **kwargs)

        if bg_color is not None:
            box.config(background=bg_color)

        if ttk.Style().theme_use() == 'clam':
            box.config(selectbackground=dict(ttk.Style().map('Toolbutton', 'background'))['active'])

        box.grid(row=0, column=0, sticky='nsew')

        scrollb = ttk.Scrollbar(self, command=box.yview)
        scrollb.grid(row=0, column=1, sticky='nsew', padx=(0, 0))
        box['yscrollcommand'] = scrollb.set

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        if items is not None:
            for item in items:
                box.insert(tk.END, item)

        box.bind("<<ListboxSelect>>", self.__select)  # (2)

        self._box = box
        self._current = None

    def listbox(self):
        return self._box

    def __select(self, event):
        self._current = [self._box.get(int(i)) for i in self._box.curselection()]

    def selection(self):
        return self._current


class TextFrame(BaseFrame):

    def __init__(self, master, bg_color=None, editable=False, scrollable=True, padding=5, *args, **kwargs):

        BaseFrame.__init__(self, master, padding=padding)

        relief = self['relief']
        if bg_color == ttk.Style().lookup('TFrame', 'background'):
            relief = tk.FLAT

        txt = tk.Text(self, borderwidth=self['borderwidth'], relief=relief, highlightthickness=0, *args, **kwargs)
        txt.grid(row=0, column=0, sticky='news')

        if bg_color is not None:
            txt.config(bg=bg_color)

        if not editable:
            txt.config(state=tk.DISABLED)

        if scrollable is not None:
            if type(scrollable) not in (tuple, list):
                scrollable = scrollable,
            if len(scrollable) > 0 and scrollable[0]:
                scrollb = ttk.Scrollbar(self, command=txt.yview, orient=tk.VERTICAL)
                scrollb.grid(row=0, column=1, sticky='news', padx=(0, 0))
                txt['yscrollcommand'] = scrollb.set
            if len(scrollable) > 1 and scrollable[1]:
                scrollb = ttk.Scrollbar(self, command=txt.xview, orient=tk.HORIZONTAL)
                scrollb.grid(row=1, column=0, sticky='news', padx=(0, 0))
                txt['xscrollcommand'] = scrollb.set

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self._text = txt
        self._editable = editable

    def text(self):
        return self._text

    def clear(self):
        self._text.config(state=tk.NORMAL)
        self._text.delete(1.0, tk.END)
        if not self._editable:
            self._text.config(state=tk.DISABLED)

    def publish(self, msg, where=tk.END):
        self._text.config(state=tk.NORMAL)
        self._text.insert(where, msg + '\n')
        if not self._editable:
            self._text.config(state=tk.DISABLED)


class TabFrame(BaseFrame):

    __widget_cfg = {'padx': 1, 'pady': 1, 'ipadx': 4, 'ipady': 0}

    def __init__(self, master, side=tk.LEFT, switch=None):
        BaseFrame.__init__(self, master)

        self.__switch = switch

        ttk.Style().configure(type(self).__name__ + '.Toolbutton', anchor='center', padding=2, relief=tk.GROOVE)
        if ttk.Style().theme_use() == 'clam':
            ttk.Style().map(type(self).__name__ + '.Toolbutton',
                        background=[('selected', dict(ttk.Style().map('Toolbutton', 'background'))['active'])])
        self.__current_frame = None
        self.__count = 0
        self.__frame_choice = tk.IntVar(0)

        if side in (tk.TOP, tk.BOTTOM):
            self.__side = tk.LEFT
        else:
            self.__side = tk.TOP

        self.__options_frame = BaseFrame(self)
        self.__options_frame.pack(side=side, fill=tk.BOTH, expand=False, **TabFrame.__widget_cfg)

        self.pack(fill=tk.BOTH, expand=True, **TabFrame.__widget_cfg)

        self.__btns = {}
        self.__max_width = 0

    def add(self, title, fr):
        b = ttk.Radiobutton(self.__options_frame, text=title, style=type(self).__name__ + '.Toolbutton', \
                           variable=self.__frame_choice, value=self.__count, \
                           command=lambda: self.select(fr))
        b.pack(fill=tk.BOTH, side=self.__side, expand=False, **TabFrame.__widget_cfg)

        self.__btns[fr] = b

        if not self.__current_frame:
            self.__current_frame = fr
            self.select(fr)
        else:
            fr.forget()

        self.__count += 1

        if len(title) > self.__max_width: self.__max_width = len(title)
        [item.config(width=self.__max_width) for key, item in self.__btns.items()]

        return b

    def select(self, fr):
        for btn in self.__btns.values(): btn.state(['!selected'])
        if self.__switch is not None: self.__switch(self.__current_frame, fr)
        self.__current_frame.forget()
        fr.pack(fill=tk.BOTH, expand=True, **TabFrame.__widget_cfg)
        self.__btns[fr].state(['selected'])
        self.__current_frame = fr


class InfoLabel(ttk.Label):

    def __init__(self, *args, **kwargs):
        ttk.Label.__init__(self, *args, **kwargs)
        self.after_task = None
        self.__animate()

    def publish(self, msg, delay=2500):
        if self.after_task is not None:
            self.after_cancel(self.after_task)
        self.config(text=msg)
        ref = self.after(delay, lambda: self.config(text=' '))
        self.after_task = ref

    def __animate(self):
        if len(self['text']) == 1:
            foo = ('\\', '|', '/', '-')
            try:
                idx = foo.index(self['text'])
            except ValueError:
                idx = -1
            self['text'] = foo[(idx+1)%len(foo)]
        self.after(500, self.__animate)

# ----------------------------

class ToolTip:

    def __init__(self, widget, text='', wrap=320):
        self.waittime = 250
        self.wraplength = wrap
        self.widget = widget
        self.text = text
        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.leave)
        self.widget.bind("<ButtonPress>", self.leave)
        self.id = None
        self.tw = None

    def enter(self, *_):
        self.schedule()

    def leave(self, *_):
        self.unschedule()
        self.hide()

    def schedule(self):
        self.unschedule()
        self.id = self.widget.after(self.waittime, self.show)

    def unschedule(self):
        _id = self.id
        self.id = None
        if _id:
            self.widget.after_cancel(_id)

    def show(self, *_):
        x, y, cx, cy = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20
        # creates a toplevel window
        self.tw = tk.Toplevel(self.widget)
        # Leaves only the label and removes the ui window
        self.tw.wm_overrideredirect(True)
        self.tw.wm_geometry("+%d+%d" % (x, y))
        label = tk.Label(self.tw, text=self.text, justify='left',
                       background="#ffffff", relief='solid', borderwidth=1,
                       wraplength = self.wraplength)
        label.pack(ipadx=1)

    def hide(self):
        tw = self.tw
        self.tw= None
        if tw:
            tw.destroy()

# ----------------------------

class Sidebar(ttk.Frame):

    def __init__(self, master, width=None):
        ttk.Frame.__init__(self, master)
        ttk.Style().configure(type(self).__name__ + '.Toolbutton', anchor='center', padding=2, relief=tk.GROOVE)
        if ttk.Style().theme_use() == 'clam':
            ttk.Style().map(type(self).__name__ + '.Toolbutton',
                        background=[('selected', dict(ttk.Style().map('Toolbutton', 'background'))['active'])])
        if width is not None:
            label = ttk.Label(self, text='', width=width)
            label.grid_columnconfigure(0, weight=1)
            label.grid(row=0, column=0, sticky='news')
        self.__parts = []

    def create_label(self, title=None, width=None):
        wrapper = ttk.Frame(self)
        wrapper.grid_columnconfigure(0, weight=1)
        wrapper.grid(row=len(self.__parts), column=0, sticky='news')
        label = ttk.Label(wrapper, text=title, width=width, padding=2, anchor=tk.CENTER)
        label.grid_columnconfigure(0, weight=1)
        label.grid(row=0, column=0, sticky='news')
        self.__parts.append(wrapper)
        wrapper.label = label
        return wrapper

    def create_frame(self):
        wrapper = ttk.Frame(self, padding=4)
        wrapper.grid_columnconfigure(0, weight=1)
        wrapper.grid(row=len(self.__parts), column=0, sticky='news')
        self.__parts.append(wrapper)
        return wrapper

    def create_text(self, width, height, scroll=False, *arg, **kwarg):
        wrapper = self.create_frame()
        if not scroll: width += 2  # approx. width of a scrollbar
        text = TextFrame(wrapper, bg_color=ttk.Style().lookup('TFrame', 'background'),
                         editable=True, scrollable=scroll, width=width, height=height, *arg, **kwarg)
        text.grid(row=0, column=0, sticky='news')
        if height is None:
            wrapper.grid_rowconfigure(0, weight=1)
            self.grid_rowconfigure(len(self.__parts) - 1, weight=1)
        wrapper.text = text
        return wrapper

    def create_buttons(self, items):
        wrapper = self.create_frame()
        inner = BaseFrame(wrapper, padding=(5, 5, 5, 5))
        inner.grid_columnconfigure(0, weight=1)
        inner.grid(row=0, column=0, sticky='news')
        btns = []
        max_width = 0
        for item in items:
            if len(item) > max_width: max_width = len(item)
        for idx, txt in enumerate(items):
            btn = ttk.Button(inner, text=txt, padding=2, width=max_width + 2)
            btn.grid(row=idx, column=0, sticky='ns', pady=2)
            btns.append(btn)
        wrapper.buttons = tuple(btns)
        return wrapper

    def create_radios(self, items, selected_value=None):
        wrapper = self.create_frame()
        inner = BaseFrame(wrapper, padding=(5, 5, 5, 5))
        inner.grid_columnconfigure(0, weight=1)
        inner.grid(row=0, column=0, sticky='news')
        btns = []
        max_width = 0
        option_value = tk.IntVar()
        for item in items:
            if len(item) > max_width: max_width = len(item)
        for idx, txt in enumerate(items):
            btn = ttk.Radiobutton(inner, text=txt, padding=2, width=max_width + 2, variable=option_value,
                                  value=idx, style = type(self).__name__ + '.Toolbutton')
            btn.grid(row=idx, column=0, sticky='ns', pady=2)
            btns.append(btn)
        option_value.set(selected_value)
        wrapper.buttons = tuple(btns)
        wrapper.variable = option_value
        return wrapper

    def create_checks(self, items):
        wrapper = self.create_frame()
        inner = BaseFrame(wrapper, padding=(5, 5, 5, 5))
        inner.grid_columnconfigure(0, weight=1)
        inner.grid(row=0, column=0, sticky='news')
        btns = []
        max_width = 0
        for item in items:
            if len(item) > max_width: max_width = len(item)
        for idx, txt in enumerate(items):
            check_value = tk.IntVar()
            btn = ttk.Checkbutton(inner, text=txt, padding=2, width=max_width + 2, variable=check_value,
                                  style=type(self).__name__ + '.Toolbutton')
            btn.grid(row=idx, column=0, sticky='ns', pady=2)
            btn.variable = check_value
            btns.append(btn)
        wrapper.buttons = tuple(btns)
        return wrapper

    def create_list(self, width, height, items):
        wrapper = self.create_frame()
        box = ListFrame(wrapper, items=items, bg_color=ttk.Style().lookup('TFrame', 'background'),
                        width=width, height=height)
        box.grid_columnconfigure(0, weight=1)
        box.grid(row=0, column=0, sticky='news')
        wrapper.frame = box
        return wrapper

# ----------------------------

class CartesianCanvas(tk.Canvas):

    def __init__(self, master, canvas_size, grid=False):
        self._width, self._height = canvas_size
        self.__grid = grid
        tk.Canvas.__init__(self, master, width=self._width, height=self._height, bg='#f0f2f4')
        self.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        self._objects = {'line': {}, 'point': {}, 'text': {}, 'ellipse': {}}
        self.clear()

    def to_screen(self, point):
        x, y, *_ = point
        w, h = self.winfo_width(), self.winfo_height()
        return x + w // 2, h // 2 - y

    def to_world(self, point):
        x, y = point
        w, h = self.winfo_width(), self.winfo_height()
        return x - w // 2, h // 2 - y

    def max_line(self, color, value):
        self.__max('line', color, value)

    def max_point(self, color, value):
        self.__max('point', color, value)

    def max_text(self, color, value):
        self.__max('text', color, value)

    def max_ellipse(self, color, value):
        self.__max('ellipse', color, value)

    def __max(self, class_, color_, value):
        try:
            sub = self._objects[class_][color_]
            obj = sub['objects']
            for o in obj: self.delete(o)
        except KeyError:
            pass
        self._objects[class_][color_] = {'index': 0, 'objects': [None] * value}

    def line(self, coord_, color, width):
        obj = self.create_line(coord_, fill=color, width=width, joinstyle=tk.ROUND, capstyle=tk.ROUND)
        return self.__put('line', color, obj)

    def point(self, at_, color, width):
        x, y = at_
        coord = (x - 1, y, x, y - 1, x + 1, y, x, y + 1, x - 1, y)
        obj = self.create_line(coord, fill=color, width=width, joinstyle=tk.ROUND, capstyle=tk.ROUND)
        return self.__put('point', color, obj)

    def text(self, at_, text, color):
        x, y = at_
        obj = self.create_text(x + 4, y + 4, text=text, anchor=tk.NW, fill=color)
        return self.__put('text', color, obj)

    def ellipse_alternative(self, at_, size_, color, width):
        x, y = at_
        a, b = size_
        obj = self.create_oval(x - a, y - b, x + a, y + b, outline=color, width=width)
        return self.__put('ellipse', color, obj)

    def ellipse(self, at_, size_, orientation_, color, width):
        x, y = at_
        w, h = size_
        pts = utl_.oval(x - h, y - w, x + h, y + w, orientation_)
        obj = self.create_polygon(tuple(pts), outline=color, fill='', width=width, smooth=True)
        return self.__put('ellipse', color, obj)

    def __put(self, class_, color_, id_):
        try:
            sub = self._objects[class_][color_]
            idx = sub['index']
            obj = sub['objects']
            if obj[idx] is not None: self.delete(obj[idx])
            sub['objects'][idx] = id_
            sub['index'] = (idx + 1) % len(obj)
        except KeyError:
            self._objects[class_][color_] = {'index': 0, 'objects': [id_]}
        return id_

    def clear(self):
        self.update()
        self.delete("all")
        self.__draw_cross(self.winfo_width(), self.winfo_height())
        self.__draw_frame(self.winfo_width(), self.winfo_height())

    def __draw_frame(self, width, height, color='#a7a7a7'):
        self.create_rectangle(1, 1, width - 2, height - 2, outline=color, width=3)

    def __draw_cross(self, width, height, color='#a7a7a7'):
        if self.__grid:
            self.create_line(width // 2, 0, width // 2, height, fill=color, dash=(5, 5))
            self.create_line(0, height // 2, width, height // 2, fill=color, dash=(5, 5))

# ----------------------------

class AppFrame(BaseFrame):

    def __init__(self, master, title=None, frame_size=(1080, 720), about=None):

        width, height = frame_size

        BaseFrame.__init__(self, master)
        self.root = master
        self.about_text = about

        self.root.title(title)

        ws = self.root.winfo_screenwidth()
        hs = self.root.winfo_screenheight()
        w = width
        h = height
        x = int((ws / 2) - (w / 2))
        y = int((hs / 2) - (h / 2))

        menu = tk.Menu(self.root)
        ttk.Style().configure('TMenu', background=master['background'])

        self.root.config(menu=menu)
        self.root.geometry('{}x{}+{}+{}'.format(w, h, x, y))
        self.root.minsize(width=w, height=h)

        filemenu = tk.Menu(menu, tearoff=0)
        menu.add_cascade(label="File", menu=filemenu)
        filemenu.add_command(label=" Exit ", command=self.close)

        self.lbl = InfoLabel(self, text=' ', relief=tk.GROOVE, anchor=tk.W, justify=tk.LEFT, padding=(2, 2, 2, 2))
        self.lbl.pack(fill=tk.X, side=tk.BOTTOM)

        self.root.bind('<Escape>', lambda _ = self.root : self.close())
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.pack(fill=tk.BOTH, expand=True)

        events.register(lambda sender, event, obj: self.message(obj), events='cursor:move')

    def close(self):
        self.root.destroy()
        self.root.quit()

    def message(self, msg, delay=5000):
        self.lbl.publish(msg, delay)

# ----------------------------

import numpy as np

np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)
np.set_printoptions(linewidth=150)
np.set_printoptions(formatter={'float': '{: 9.3f}'.format})

# ----------------------------

class FilterCanvas(CartesianCanvas):

    STEP_MILLIS = 10  # clock

    def __init__(self, master, size, targets, sensor_array, model_call, model_args, filter_call, filter_args):

        CartesianCanvas.__init__(self, master, size, True)
        ToolTip(master,
                'controls: left mouse button -> start/reset' + ', ' +
                'right mouse button -> pause/resume' + '\n' +
                'settings: ' + str(model_args), 600)

        self._enabled = False
        self._choice = 0

        self._model_call = model_call
        self._model_args = model_args
        self._model_instance = model_call(**model_args)

        self._dim = self._model_instance.dimension(env_.Quantity.POSITION)

        self._filter_call = filter_call
        self._filter_args = filter_args
        self._filter_instance = None

        self._filter_targets = targets

        self._net = env_.Net(self)

        self._clock = env_.Clock(self._net)  # local clock

        self._last_packet = None

        sensors = []
        for sensor_args in sensor_array:
            sen = env_.Sensor(**sensor_args, net=self._net)
            sensors.append(sen)
        self._sensors = sensors

        self._location = self.to_screen((0, 0))

        self.bind("<Button-1>", lambda event: self.__toggle(event, 0))
        self.bind("<Button-3>", lambda event: self.__toggle(event, 1))
        self.bind("<Motion>", self.__capture)

        self.__f_draw = lambda ori, evt, msg: self.__draw(msg)
        env_.events.register(self.__f_draw, events='world:signal')

        self.__f_estimate = lambda ori, evt, msg: self.__data(ori)
        env_.events.register(self.__f_estimate, events='estimate:change')

        self.__f_data_accept = lambda ori, evt, msg: self.__data_accept(msg)
        env_.events.register(self.__f_data_accept, events=('filter:accept', 'filter:apply'))

        self.__f_data_reject = lambda ori, evt, msg: self.__data_reject(msg)
        env_.events.register(self.__f_data_reject, events=('filter:reject', 'filter:drop'))

        self.__f_status = lambda ori, evt, msg: self.__filter_status(ori, msg)
        env_.events.register(self.__f_status, events='filter:status')

        self.__f_sensor = lambda ori, evt, msg: self.__sensor(msg)
        env_.events.register(self.__f_sensor, events=self._net.id('sensor:data'))  # first measurement for init of filter

        self.__trace()
        self.__loop()

    def net(self):
        return self._net

    def sensors(self):
        return self._sensors

    def __refresh(self):
        events.notify(self, event=self._net.id('refresh'), msg=self.__info())

    def __info(self):
        est = self._filter_instance.estimate()
        pro = self._filter_instance.process()
        mea = self._filter_instance.measurements()
        t, z, R, H = float('nan'), None, None, None
        if len(mea) > 0:
            t, z, R, H = mea[0].time(), mea[0].value().T, mea[0].noise(), mea[0].transition()
        _, F = pro.transition(env_.Utilization.PROCESS)
        Q = pro.noise()
        s =  ''
        s += 't: ' + 'predict: ' + str(est.time()) + ', correct: ' + str(t) + '\n\n' + \
             'x: state estimate\n' + str(est.value().T) + '\n\n' + \
             'P: state estimate covariance\n' + str(est.noise()) + '\n\n' + \
             'K: gain\n' + str(self._filter_instance.gain()) + '\n\n'
        s += 'z: observation\n' + str(z) + '\n\n' + \
             'R: observation noise\n' + str(R) + '\n\n' + \
             'H: observation transition\n' + str(H) + '\n\n'
        s += 'Q: process noise distribution\n' + str(Q) + '\n\n' + \
             'F: process transition\n' + str(F)
        return s

    def __toggle(self, event, value):
        self.__capture(event)
        if value == 0:
            if self._choice % 2 == 0:
                events.notify(self, event=self._net.id('reset'), msg=True)
                self._filter_instance = self._filter_call(
                    self._net,
                    self._model_instance,
                    **self._filter_args,
                    origin_uncertain=self._filter_targets != 1)
                events.notify(self, event=self._net.id('init'), msg=self._filter_instance)
                self._clock.reset()
                self._clock.start()
                self._enabled = True
            if self._choice % 2 == 1:
                events.notify(self, event=self._net.id('reset'), msg=False)
                self._clock.stop()
                self._clock.reset()
                self.clear()
                self._filter_instance.destroy()
                self._filter_instance = None
                events.notify(self, event=self._net.id('init'), msg=self._filter_instance)
                self.choice = 0
                self._enabled = False
                events.notify(self, event=self._net.id('refresh'), msg='')
            self._choice += 1
        elif self._choice % 2 == 1:
            self._enabled = not self._enabled
            if self._enabled:
                self._clock.start()
            else:
                self._clock.stop()

    def __capture(self, event):
        x_new, y_new = self.to_world((event.x, event.y))
        if self._dim == 2:
            self._location = x_new, y_new
        elif self._dim == 3:
            self._location = x_new, y_new, (x_new + y_new) / 2
        events.notify(self,event='cursor:move', msg='x = {:3d}  y = {:3d}  |  loc = {}  |  t = {:.2f}'
                           .format(event.x, event.y, self._location, self._clock.time() / 1000))

    def __data(self, ori):
        if self._enabled and self.winfo_viewable():
            val, _ = ori.extract(env_.Quantity.POSITION, True)
            xy = self.to_screen(val)
            self.point(xy, 'red', 3)
            _, noi = ori.extract(env_.Quantity.POSITION)
            w, h, theta = utl_.ellipse_covariance_2d(noi[:2,:2], 3)
            self.ellipse(xy, (w, h), (-theta + 360) % 360, 'red', 1)

    def __data_accept(self, msg):
        if self._enabled and self.winfo_viewable():
            value, noise = msg.value(True), msg.noise(True)
            x, y, *_ = value
            self.point(self.to_screen((x, y)), 'gray', 2)
            path = self._filter_instance.path()
            for point in path:
                self.point(self.to_screen(point), 'black', 3)

    def __data_reject(self, msg):
        if self._enabled and self.winfo_viewable():
            value, noise = msg.value(True), msg.noise(True)
            x, y, *_ = value
            self.point(self.to_screen((x, y)), 'lightgray', 2)

    def __filter_status(self, ori, msg):
        if ori == self._filter_instance:
            pass

    def __loop(self):
        self._clock.tick(FilterCanvas.STEP_MILLIS)
        self.after(FilterCanvas.STEP_MILLIS, lambda: self.__loop())
        if self._enabled: self.__refresh()

    def __trace(self, interval=10):
        self.after(interval, lambda: self.__trace(interval))

        if self._enabled and self.winfo_viewable():
            arr = []
            if self._dim == 2:

                xnew, ynew = self._location
                if self._filter_targets == 1:
                    arr = [(xnew, ynew)]
                elif self._filter_targets == 2:
                    arr = [(xnew, ynew), (ynew, xnew)]
                elif self._filter_targets == 3:
                    arr = [(xnew, ynew), (ynew, xnew), (abs(xnew), abs(ynew))]
                elif self._filter_targets == 4:
                    arr = [(xnew, ynew), (ynew, xnew), (abs(xnew), abs(ynew)), (-abs(xnew), -abs(ynew))]
            elif self._dim == 3:

                xnew, ynew, znew = self._location
                if self._filter_targets == 1:
                    arr = [(xnew, ynew, znew)]
                elif self._filter_targets == 2:
                    arr = [(xnew, ynew, znew), (ynew, xnew, znew)]
                elif self._filter_targets == 3:
                    arr = [(xnew, ynew, znew), (ynew, xnew, znew), (abs(xnew), abs(ynew), znew)]
                elif self._filter_targets == 4:
                    arr = [(xnew, ynew, znew), (ynew, xnew, znew), (abs(xnew), abs(ynew), znew), (-abs(xnew), -abs(ynew), znew)]

            env_.events.notify(self, event='target:signal', msg=(self._clock.time(), self._filter_instance, arr[0]))  # used only for monitoring! to do before shuffeling!

            random.shuffle(arr)  # randomize order of signals
            for arg in arr:
                env_.events.notify(self, event='world:signal', msg=(env_.Quantity.POSITION, arg))  # regular signal from environment collectable by all sensors

    def __draw(self, msg):
        if self._enabled and self.winfo_viewable():
            q, p = msg
            if q == env_.Quantity.POSITION:
                x, y = self.to_screen(p)
                self.point((x, y), '#b7b7b7', 1)

    def __sensor(self, msg):
        if self._enabled and self.winfo_viewable():
            if utl_.euclidean_distance(msg.value()[:2], self._location) < 50:  # measurements close to mouse cursor are valid
                if self._filter_instance is not None and \
                        not self._filter_instance.initialize():
                    self._filter_instance.initialize(msg, 50, 1000)  # initial gate size is 50 pixels

# ----------------------------

class ControlFrame(Sidebar):

    def __init__(self, master, canvas, width=64):

        Sidebar.__init__(self, master)
        self.grid(row=0, column=0, padx=5, pady=5, sticky='news')

        self._filter_instance = None

        self._canvas = canvas
        self._canvas.max_ellipse('red', 1)
        self._canvas.max_point('red', 125)
        self._canvas.max_point('#b7b7b7', 275)
        self._canvas.max_point('green', 50)
        self._canvas.max_point('orange', 50)
        self._canvas.max_point('gray', 50)
        self._canvas.max_point('lightgray', 50)
        self._canvas.max_point('black', 2)

        self._sensors = canvas.sensors()

        sensor_label = []
        for sensor in self._sensors:
            sensor_label.append(str(sensor) +
                                " noise=" + str(sensor.noise()).replace(' ', '')[1:-1] +
                                " bias=" + str(sensor.bias()).replace(' ', '')[1:-1] +
                                " interval=" + str(sensor.interval()))

        self.create_label(':: SENSOR ARRAY ::')
        self._sensor_button_frame = self.create_checks(sensor_label)
        for i, btn in enumerate(self._sensor_button_frame.buttons):
            btn.config(command=lambda btn=btn, i=i:self.__toggle_sensor(btn, i))
            if i<2: btn.variable.set(int(True))

        self.create_label(':: FILTER DETAILS ::')
        self._info = self.create_text(width, None, (True, True), wrap='none')

        events.register(lambda sender, event, msg: self.__init_filter(msg), events=canvas.net().id('init'))
        events.register(lambda sender, event, msg: self.__reset_filter(msg), events=canvas.net().id('reset'))
        events.register(lambda sender, event, msg: self.__refresh_text(msg), events=canvas.net().id('refresh'))

    def __toggle_sensor(self, button, index):
        if self._filter_instance is not None:
            self._sensors[index].active(bool(button.variable.get()))

    def __refresh_text(self, msg):
        self._info.text.clear()
        self._info.text.publish(msg)

    def __reset_filter(self, msg):
        if msg:
            for i, btn in enumerate(self._sensor_button_frame.buttons):
                self._sensors[i].active(bool(btn.variable.get()))
        else:
            for i, btn in enumerate(self._sensor_button_frame.buttons):
                self._sensors[i].active(False)

    def __init_filter(self, filter_):
        self._filter_instance = filter_

# ----------------------------

class FilterFrame(BaseFrame):

    def __init__(self, master, targets, sensors, model_call, model_args, filter_call, filter_args):
        BaseFrame.__init__(self, master)

        self._cnv_wrapper = tk.Frame(self)
        self._cnv_frame = BaseFrame(self._cnv_wrapper)
        self._cnv = FilterCanvas(self._cnv_frame, (600, 600), targets, sensors, model_call, model_args, filter_call, filter_args)
        self._cnv_frame.pack(fill=tk.BOTH, expand=True)
        self._cnv_wrapper.grid(column=0, row=0, sticky='news')

        self._sidebar_wrapper = BaseFrame(self)
        self._sidebar_frame = ControlFrame(self._sidebar_wrapper, self._cnv)
        self._sidebar_wrapper.grid_rowconfigure(0, weight=1)
        self._sidebar_wrapper.grid(column=1, row=0, sticky='news')

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

