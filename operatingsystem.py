import threading
import time
import random


class Menu:
    def __init__(self):
        self.items = {
            "coffee": {"price": 3, "time": 2},
            "tea": {"price": 2, "time": 1},
            "croissant": {"price": 4, "time": 3},
            "sandwich": {"price": 5, "time": 4}
        }

    def get_item(self, item_name):
        return self.items.get(item_name, None)


class Order:
    def __init__(self, customer, items):
        self.customer = customer
        self.items = items
        self.status = "wait"
        self.estimated_time = sum([item['time'] for item in items])

    def total_price(self):
        return sum([item['price'] for item in self.items])


class Customer:
    def __init__(self, name):
        self.name = name

    def notify(self, order):
        print(f"[Notification]:  '{self.name}', your order is ready. the total: ${order.total_price()}")


class Barista(threading.Thread):
    def __init__(self, name):
        threading.Thread.__init__(self)
        self.name = name
        self.current_order = None
        self.lock = threading.Lock()

    def run(self):
        while True:
            if self.current_order:
                with self.lock:
                    self.current_order.status = "in progress"
                    print(f"Barista '{self.name}' is preparing order for '{self.current_order.customer.name}'.")
                    time.sleep(self.current_order.estimated_time)
                    self.current_order.status = "completed"
                    print(f"Barista '{self.name}' completed order for '{self.current_order.customer.name}'.")
                    self.current_order.customer.notify(self.current_order)
                    self.current_order = None


class CoffeeShop:
    def __init__(self, num_baristas):
        self.menu = Menu()
        self.baristas = [Barista(f"Barista-{i + 1}") for i in range(num_baristas)]
        for barista in self.baristas:
            barista.start()

    def place_order(self, customer, item_names):
        items = []
        for name in item_names:
            item = self.menu.get_item(name)
            if item:
                items.append(item)

        order = Order(customer, items)

        assigned = False
        while not assigned:
            for barista in self.baristas:
                if not barista.current_order:
                    with barista.lock:
                        barista.current_order = order
                        print(f"Order from '{customer.name}' assigned to '{barista.name}'.")
                        assigned = True
                        break
            if not assigned:
                print(f"No barista available for '{customer.name}', waiting")
                time.sleep(1)


if __name__ == "__main__":
    shop = CoffeeShop(num_baristas=2)

    customers = [Customer("evdo"), Customer("thomas"), Customer("danae")]
    orders = [["coffee", "croissant"], ["tea", "sandwich"], ["coffee", "tea", "sandwich"]]

    for customer, items in zip(customers, orders):
        threading.Thread(target=shop.place_order, args=(customer, items)).start()