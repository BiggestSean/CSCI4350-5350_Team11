#Todo:
#May need to create disincentive for price increases. If
#Add Vegetarian foods to menu, Vegetarians won't be able to pick meat-based options


from dataclasses import dataclass, field
from collections import deque
from datetime import datetime, timedelta
import random

#Open for 10 hours (2-12AM), but stops seating customers durnig the last hour (11PM)
#Code simulates 9 hours with 5 minute timing intervals 
STEP = 5
OPEN_MINUTE = 14 * 60
CLOSE_MINUTE = 23 * 60
CAPACITY = 100


@dataclass
class Dish:
    dish_name: str
    primary_ingredient: str
    cost_to_make: float
    regular_price: float
    preparation_minutes: int


class Menu:

    def __init__(self): # "Dish Name", "Ingredient", cost_to_make, price, prep_time
        self.dishes = [
            Dish("Porkchop", "Pork", 8.00, 21.00, 25),
            Dish("Chicken Alfredo", "Chicken", 6.50, 19.00, 20),
            Dish("Shrimp Scampi", "Shrimp", 9.00, 25.00, 20),
            Dish("Ribeye Steak", "Steak", 17.00, 39.00, 30),

        ]

    def price(self, dish): #--------------------------------------------------------------------------flex pricing code goes in this function.

        
        return dish.regular_price  

    def choose_dish(self, customer, rng): #Function for setting a customer's dish preference
                                        # Favorite ingredients are weighted higher; expensive dishes weighted lower.
        weights = [
            (3.0 if d.primary_ingredient == customer.preferred_food else 1.0)
            * (20.0 / self.price(d))
            for d in self.dishes
        ]
        return rng.choices(self.dishes, weights=weights, k=1)[0]



    
#customer class, sets food preferences, waiting time tolerance (patience_level), and the wait times for that individual
@dataclass 
class Customer:
    customer_id: int
    preferred_food: str
    patience_level: int
    arrival_minute: int
    status: str = "waiting_for_seat"
    seated_minute: int | None = None
    order_minute: int | None = None
    food_ready_minute: int | None = None
    finish_minute: int | None = None
    dish: Dish | None = None


    
#metrics tracking class
@dataclass
class Stats:
    arrived: int = 0
    served: int = 0
    left_waiting_for_seat: int = 0
    left_waiting_for_food: int = 0
    revenue: float = 0.0
    food_cost: float = 0.0
    max_occupied: int = 0
    max_queue: int = 0



    
#Restaurant class holds menu, Customers and where the Customers currently are (waiting or seated)
class Restaurant:
    def __init__(self, seed=42):
        self.rng = random.Random(seed)
        self.menu = Menu()
        self.waiting = deque()
        self.seated = []
        self.stats = Stats()
        self.next_customer_id = 1

    @staticmethod
    def clock(minute):
        return f"{minute // 60:02d}:{minute % 60:02d}"

    @staticmethod
    def demand(minute): 
        """Customer arrival volume increases from 2 PM to 6 PM, peaks 6-8PM, then tapers off."""
        if minute < 18 * 60:
            return 0.2 + 0.8 * (minute - OPEN_MINUTE) / (4 * 60)
        if minute < 20 * 60:
            return 1.0
        # Slowly taper after 8 PM through 11 PM.
        return 1.0 - 0.65 * (minute - 20 * 60) / (3 * 60)

    def incoming_customers(self, minute): 
        # Exactly 0-8 new customers each interval; more guests at peak.
        n = sum(self.rng.random() < self.demand(minute) for _ in range(8))
        for _ in range(n):
            c = Customer(                                      #Generate new customers
                customer_id=self.next_customer_id,
                preferred_food=self.rng.choice(["Pork", "Chicken", "Shrimp", "Steak"]),
                patience_level=self.rng.randint(0, 6),
                arrival_minute=minute,
            )
            self.next_customer_id += 1
            self.waiting.append(c)
        self.stats.arrived += n
        return n

    def seat_waiting_customers(self, minute):
        while self.waiting and len(self.seated) < CAPACITY:
            c = self.waiting.popleft()
            c.status = "deciding"
            c.seated_minute = minute
            # Possible delays 0, 5, 10, or 15 minutes, 5 being most likely.
            delay = self.rng.choices([0, 5, 10, 15], weights=[1, 6, 3, 1])[0]
            c.order_minute = minute + delay
            self.seated.append(c)

    def process_waiting(self, minute): #Handles waiting customers
        remaining = deque()
        for c in self.waiting:
            # Patience 0: 5 minutes, patience 5: 30 minutes.
            limit = STEP * (c.patience_level + 1)
            if minute - c.arrival_minute >= limit:
                c.status = "left_without_seat"
                self.stats.left_waiting_for_seat += 1
            else:
                remaining.append(c)
        self.waiting = remaining

    def process_seated(self, minute):
        remaining = []
        for c in self.seated:
            if c.status == "deciding" and minute >= c.order_minute:
                c.dish = self.menu.choose_dish(c, self.rng)
                c.status = "waiting_for_food"
                c.food_ready_minute = minute + c.dish.preparation_minutes
                # Preparation cost occurs when the kitchen begins an order.
                self.stats.food_cost += c.dish.cost_to_make

            if c.status == "waiting_for_food":
                # Food wait tolerance: 15 to 65 minutes by patience level.
                food_limit = 5 + 10 * c.patience_level
                if minute >= c.food_ready_minute:
                    c.status = "eating"
                    meal_minutes = self.rng.choices(
                        list(range(30, 91, 5)),
                        weights=[max(1, 13 - abs(m - 45) / 5) for m in range(30, 91, 5)],
                    )[0]
                    c.finish_minute = minute + meal_minutes

                #Customer much less likely to leave after being seated, but still possible
            else:
                wait_time = minute - c.order_minute

                if wait_time < 30:
                    leave_chance = 0.01
                elif wait_time < 40:
                    leave_chance = 0.05
                elif wait_time < 50:
                    leave_chance = 0.15
                elif wait_time < 60:
                    leave_chance = 0.30
                else:
                    leave_chance = 0.50

                # More patient customers are less likely to leave
                leave_chance = leave_chance / (1 + c.patience_level * 0.3)

                if self.rng.random() < leave_chance:
                     c.status = "left_waiting_for_food"
                     self.stats.left_waiting_for_food += 1
                     continue
    
#            elif minute - c.order_minute >= food_limit:
#                    c.status = "left_waiting_for_food"
#                    self.stats.left_waiting_for_food += 1
#                    continue

            if c.status == "eating" and minute >= c.finish_minute:   #Record served and revenue stats
                c.status = "paid_and_left"
                self.stats.served += 1
                self.stats.revenue += self.menu.price(c.dish)
                continue

            remaining.append(c)
        self.seated = remaining

    def step(self, minute):
        # Process existing customers, then arrivals, then fill empty seats or wait.
        self.process_seated(minute)
        self.process_waiting(minute)
        arrivals = self.incoming_customers(minute) if minute < CLOSE_MINUTE else 0
        self.seat_waiting_customers(minute)
        self.stats.max_occupied = max(self.stats.max_occupied, len(self.seated))
        self.stats.max_queue = max(self.stats.max_queue, len(self.waiting))
        return arrivals

    def run(self, verbose=False):
        minute = OPEN_MINUTE
        # After 11 PM, stop accepting arrivals, but finish current service.
        while minute < CLOSE_MINUTE or self.waiting or self.seated:
            arrivals = self.step(minute)
            if verbose:
                print(f"{self.clock(minute)} | New: {arrivals} | "
                    f"Seated: {len(self.seated)} | Waiting: {len(self.waiting)} | "
                    f"Paid: {self.stats.served} | "
                    f"Left for Seat: {self.stats.left_waiting_for_seat} | "
                    f"Left for Food: {self.stats.left_waiting_for_food} | "
                    f"Total Left: {self.stats.left_waiting_for_seat + self.stats.left_waiting_for_food}")
            minute += STEP
        return self.stats





if __name__ == "__main__":
    restaurant = Restaurant(seed=42)  # Set RNG Seed.
    result = restaurant.run(verbose=True)  # Set to True to see every 5-minute step.
    print("RESTAURANT SIMULATION RESULTS")





















