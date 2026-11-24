#Todo (done):
#Disincentive for price increases: see PRICE_INCREASE_PENALTY in RestaurantEnv.step
#Vegetarian foods added; vegetarian customers only pick vegetarian dishes


from dataclasses import dataclass, field
from collections import deque
from datetime import datetime, timedelta
import random

#Open for 12 hours (2-12AM), but stops seating customers during the last hour (11PM)
#Code simulates 9 hours with 5 minute timing intervals 
STEP = 5
OPEN_MINUTE = 14 * 60
CLOSE_MINUTE = 23 * 60
CAPACITY = 100

# RL: settings added for the Q-learning agent
COOKS = 17                              # RL: kitchen can cook this many orders at once (busy enough that discounts matter)
QUICK_PREP_MINUTES = 15                 # RL: dishes this fast get discounted
DISCOUNTS = (0.00, 0.10, 0.20, 0.25)    # RL: the agent's 4 actions
SATISFACTION_BASELINE = 4.0             # RL: wait ratings below this are penalized
SATISFACTION_PENALTY = 8.0              # RL: per customer, per rating point below baseline
WALKOUT_PENALTY = 25.0                  # RL: more than the profit on any dish
PRICE_INCREASE_PENALTY = 5.0            # Todo: cost per discount level removed (prices going up)
VEGETARIAN_SHARE = 0.15                 # Todo: share of customers who are vegetarian


@dataclass
class Dish:
    dish_name: str
    primary_ingredient: str
    cost_to_make: float
    regular_price: float
    preparation_minutes: int
    vegetarian: bool = False  # Todo


class Menu:

    def __init__(self): # "Dish Name", "Ingredient", cost_to_make, price, prep_time
        self.dishes = [
            # Vegetarian
            Dish("Garden Salad", "Greens", 3.50, 12.00, 10, vegetarian=True),
            Dish("Pasta Primavera", "Vegetables", 4.50, 16.00, 15, vegetarian=True),
            Dish("Four Cheese Mac and Cheese", "Vegetables", 4.50, 14.00, 15, vegetarian=True),
            Dish("Black bean Burger", "Vegetables", 6.50, 17.00, 20, vegetarian=True),
            Dish("Cauliflower Wings", "Vegetables", 4.00, 13.00, 15, vegetarian=True),
            Dish("Vegetable Fajitas", "Vegetables", 5.50, 17.00, 25, vegetarian=True),
            Dish("Margherita Flatbread", "Greens", 4.50, 15.00, 15, vegetarian=True),
            Dish("Vegetarian Burrito Bowl", "Vegetables", 4.50, 15.00, 15, vegetarian=True),

            # Pork
            Dish("Porkchop", "Pork", 8.00, 21.00, 25),
            Dish("Baby Back Ribs", "Pork", 12.00, 29.00, 25),

            # Beef
            Dish("Ribeye Steak", "Steak", 17.00, 39.00, 30),
            Dish("8oz Sirloin", "Steak", 10.00, 26.00, 25),
            Dish("6oz Sirloin", "Steak", 7.50, 21.00, 20),
            Dish("Hamburger", "Beef", 4.50, 14.00, 15),
            Dish("Cheeseburger", "Beef", 5.00, 15.00, 15),

            # Chicken
            Dish("Chicken Alfredo", "Chicken", 6.50, 19.00, 20),
            Dish("Chicken Sandwich", "Chicken", 4.50, 14.00, 15),
            Dish("Grilled Chicken Sandwich", "Chicken", 4.75, 15.00, 20),
            Dish("Fajita Wrap", "Chicken", 5.50, 16.00, 15),
            Dish("Grilled Chicken Breast", "Chicken", 6.00, 18.00, 20),
            Dish("Chicken Tenders", "Chicken", 5.00, 15.00, 15),
            Dish("Broccoli Chicken Alfredo", "Chicken", 7.25, 21.00, 20),

            # Fish and Seafood
            Dish("Shrimp Scampi", "Fish", 9.00, 25.00, 20),
            Dish("Shrimp Alfredo", "Fish", 9.50, 26.00, 20),
            Dish("Blackened Tilapia", "Fish", 6.50, 19.00, 20),
            Dish("Sesame Salmon", "Fish", 11.00, 28.00, 25),
            Dish("Hand Battered Fish", "Fish", 6.00, 18.00, 15),
        ]
        
        self.discount_level = 0  # RL


    def set_discount(self, action):  # RL: the agent picks one of DISCOUNTS
        self.discount_level = action

    def price(self, dish): #--------------------------------------------------------------------------flex pricing code goes in this function.
        # RL: discount only the quick dishes (Chicken Alfredo, Shrimp Scampi)
        if dish.preparation_minutes <= QUICK_PREP_MINUTES:
            return round(dish.regular_price * (1 - DISCOUNTS[self.discount_level]), 2)
        return dish.regular_price

    def choose_dish(self, customer, rng): #Function for setting a customer's dish preference
                                        # Favorite ingredients are weighted higher; expensive dishes weighted lower.
        # Todo: vegetarians can only pick vegetarian dishes
        options = [d for d in self.dishes if d.vegetarian or not customer.vegetarian]
        weights = [
            (3.0 if d.primary_ingredient == customer.preferred_food else 1.0)
            * (20.0 / self.price(d))
            for d in options
        ]
        return rng.choices(options, weights=weights, k=1)[0]



    
#customer class, sets food preferences, waiting time tolerance (patience_level), and the wait times for that individual
@dataclass 
class Customer:
    customer_id: int
    preferred_food: str
    patience_level: int
    arrival_minute: int
    vegetarian: bool = False  # Todo
    status: str = "waiting_for_seat"
    seated_minute: int | None = None
    order_minute: int | None = None
    food_ready_minute: int | None = None
    finish_minute: int | None = None
    dish: Dish | None = None
    price_paid: float = 0.0        # RL: price locked in when ordering
    cook_start: int | None = None  # RL: when a cook starts the order


    
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
    total_wait: float = 0.0                     # RL
    recent_waits: deque = field(default_factory=lambda: deque(maxlen=20))  # RL
    feedback: list = field(default_factory=list)  # RL: one entry per customer who left

    # RL: names q_agent.py expects
    @property
    def profit(self):
        return self.revenue - self.food_cost

    @property
    def left_seat(self):
        return self.left_waiting_for_seat

    @property
    def left_food(self):
        return self.left_waiting_for_food

    @property
    def average_customer_wait(self):
        return self.total_wait / len(self.feedback) if self.feedback else 0.0

    @property
    def average_satisfaction(self):
        return sum(f["wait_rating"] for f in self.feedback) / len(self.feedback) if self.feedback else 0.0

    @property
    def recent_wait(self):
        return sum(self.recent_waits) / len(self.recent_waits) if self.recent_waits else 0.0



    
#Restaurant class holds menu, Customers and where the Customers currently are (waiting or seated)
class Restaurant:
    def __init__(self, seed=42):
        self.rng = random.Random(seed)
        self.menu = Menu()
        self.waiting = deque()
        self.seated = []
        self.stats = Stats()
        self.next_customer_id = 1
        #self.cook_free = [OPEN_MINUTE] * COOKS  # RL: when each cook is next free
        self.kitchen_orders = deque()
        self.active_orders = {}
        '''
    def kitchen_queue(self, minute):  # RL: orders placed but not started yet
        return sum(1 for c in self.seated
                   if c.status == "waiting_for_food" and c.cook_start > minute)
'''
    def kitchen_queue(self, minute):
        return len(self.kitchen_orders)

        
    def record_exit(self, c, minute, reason):  # RL: wait-based 1-5 rating for every customer
        wait = minute - c.arrival_minute if reason != "served" else c.food_ready_minute - c.arrival_minute
        rating = 1.0 if reason != "served" else max(1.0, min(5.0, 5 - 3 * wait / 60))
        self.stats.total_wait += wait
        self.stats.recent_waits.append(wait)
        self.stats.feedback.append({"reason": reason, "wait": wait, "wait_rating": rating})

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
            vegetarian = self.rng.random() < VEGETARIAN_SHARE  # Todo
            foods = ["Vegetables", "Greens"] if vegetarian else ["Pork", "Chicken", "fish", "Steak", "Vegetables", "Greens"]
            c = Customer(                                      #Generate new customers
                customer_id=self.next_customer_id,
                preferred_food=self.rng.choice(foods),
                patience_level=self.rng.randint(0, 6),
                arrival_minute=minute,
                vegetarian=vegetarian,
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
                self.record_exit(c, minute, "left_seat")  # RL
            else:
                remaining.append(c)
        self.waiting = remaining

    def process_seated(self, minute):
        remaining = []
        for c in self.seated:
            '''
            if c.status == "deciding" and minute >= c.order_minute:
                c.dish = self.menu.choose_dish(c, self.rng)
                c.status = "waiting_for_food"
                c.price_paid = self.menu.price(c.dish)  # RL: price at time of order
                cook = min(range(COOKS), key=lambda i: self.cook_free[i])  # RL: next free cook
                c.cook_start = max(minute, self.cook_free[cook])
                c.food_ready_minute = c.cook_start + c.dish.preparation_minutes
                self.cook_free[cook] = c.food_ready_minute
                # Preparation cost occurs when the kitchen begins an order.
                self.stats.food_cost += c.dish.cost_to_make
'''
            if c.status == "deciding" and minute >= c.order_minute:
                c.dish = self.menu.choose_dish(c, self.rng)
                c.status = "waiting_for_food"
                c.price_paid = self.menu.price(c.dish)
                c.order_minute = minute

                self.kitchen_orders.append(c)

                
            if c.status == "waiting_for_food":
                # Food wait tolerance: 15 to 65 minutes by patience level.
                #food_limit = 5 + 10 * c.patience_level
                if c.food_ready_minute is not None and minute >= c.food_ready_minute:
                    c.status = "eating"
                    meal_minutes = self.rng.choices(
                        list(range(30, 91, 5)),
                        weights=[max(1, 13 - abs(m - 45) / 5) for m in range(30, 91, 5)],
                    )[0]
                    c.finish_minute = minute + meal_minutes

                #Customer much less likely to leave after being seated, but still possible
                else:  # RL: fixed indentation; this used to run for eating customers too
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

                        self.cancel_order(c)
                        
                        self.record_exit(c, minute, "left_food")  # RL
                        continue
    
#            elif minute - c.order_minute >= food_limit:
#                    c.status = "left_waiting_for_food"
#                    self.stats.left_waiting_for_food += 1
#                    continue

            if c.status == "eating" and minute >= c.finish_minute:   #Record served and revenue stats
                c.status = "paid_and_left"
                self.stats.served += 1
                self.stats.revenue += c.price_paid  # RL: was the current price, which may have changed
                self.record_exit(c, minute, "served")  # RL
                continue

            remaining.append(c)
        self.seated = remaining

    def step(self, minute):
        # Process existing customers, then arrivals, then fill empty seats or wait.
        self.process_seated(minute)
        self.process_kitchen(minute)
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
                    f"Seated: {len(self.seated):3d} | Waiting: {len(self.waiting):2d} | "
                    f"Paid: {self.stats.served:3d} | "
                    f"Left for Seat: {self.stats.left_waiting_for_seat:3d} | "
                    f"Left for Food: {self.stats.left_waiting_for_food:3d} | "
                    f"Total Left: {(self.stats.left_waiting_for_seat + self.stats.left_waiting_for_food):3d} | "
                    f"Kitchen Queue: {len(self.kitchen_orders):2d} | "
                    f"Active Cooks: {len(self.active_orders):2d}/{COOKS} | "
                    f"Profit: ${self.stats.revenue - self.stats.food_cost:,.2f}")
            minute += STEP
        return self.stats

    def process_kitchen(self, minute): #------------------------------------------------kitchen experiment

        # Release cooks who have finished their orders.
        for cook, c in list(self.active_orders.items()):
            if minute >= c.food_ready_minute:
                del self.active_orders[cook]

        # Assign waiting orders to available cooks.
        for cook in range(COOKS):

            if not self.kitchen_orders:
                break

            if cook not in self.active_orders:
                c = self.kitchen_orders.popleft()

                c.cook_start = minute
                c.food_ready_minute = (
                    minute + c.dish.preparation_minutes
                )

                self.active_orders[cook] = c

                # Food cost occurs when preparation begins.
                self.stats.food_cost += c.dish.cost_to_make

    def cancel_order(self, customer):

        # Cancel an order that has not started.
        if customer in self.kitchen_orders:
            self.kitchen_orders.remove(customer)
            return

        # Cancel an order currently being prepared.
        for cook, c in list(self.active_orders.items()):
            if c.customer_id == customer.customer_id:
                del self.active_orders[cook]
                return

# RL: ENVIRONMENT FOR q_agent.py

class RestaurantEnv:
    """One simulated day. Actions: index into DISCOUNTS. Same interface as V2/V3."""

    OBSERVATION_LABELS = ("time", "recent_wait", "kitchen_queue", "discount_level")

    def __init__(self, seed=42):
        self.seed = seed
        self.reset()

    def reset(self, seed=None):
        if seed is not None:
            self.seed = seed
        self.restaurant = Restaurant(self.seed)
        self.seed += 1
        self.minute = OPEN_MINUTE
        self.done = False
        return self.observation(), {}

    def observation(self):
        r = self.restaurant
        return (
            min(1.0, (self.minute - OPEN_MINUTE) / (CLOSE_MINUTE - OPEN_MINUTE)),
            # Scaled so q_agent.py's buckets fit V1's longer waits and queues:
            # wait buckets become <42 / 42-52 / >52 min, queue buckets <10 / 10-34 / >34 orders.
            min(1.5, r.stats.recent_wait / 180),
            r.kitchen_queue(self.minute) / 180,
            r.menu.discount_level / (len(DISCOUNTS) - 1),
        )

    def step(self, action):
        r = self.restaurant
        old_profit, old_count = r.stats.profit, len(r.stats.feedback)
        price_increase = 0.0
        if self.minute < CLOSE_MINUTE:
            # Todo: discourage raising prices (removing a discount). Customers notice
            # price jumps, and this also stops the agent flip-flopping every step.
            price_increase = PRICE_INCREASE_PENALTY * max(0, r.menu.discount_level - action)
            r.menu.set_discount(action)
        r.step(self.minute)
        self.minute += STEP
        if self.minute >= CLOSE_MINUTE:  # finish the day without more decisions
            while r.waiting or r.seated:
                r.step(self.minute)
                self.minute += STEP
            self.done = True
        new = r.stats.feedback[old_count:]
        penalty = sum(SATISFACTION_PENALTY * max(0.0, SATISFACTION_BASELINE - f["wait_rating"]) for f in new)
        penalty += WALKOUT_PENALTY * sum(f["reason"] != "served" for f in new)
        reward = (r.stats.profit - old_profit) - penalty - price_increase
        return self.observation(), reward, self.done, False, {}


if __name__ == "__main__":
    restaurant = Restaurant(seed=None)  # Set RNG Seed.
    result = restaurant.run(verbose=True)  # Set to True to see every 5-minute step.
    print("RESTAURANT SIMULATION RESULTS")