class Vehicle:
    def info(self):
        print("This is a vehicle")


class Car(Vehicle):
    def car_info(self):
        print("This is a car")


class ElectricCar(Car):
    def battery_info(self):
        print("This car has a battery")


electric_car = ElectricCar()
electric_car.info()       
electric_car.car_info()
electric_car.battery_info()   
