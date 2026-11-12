#python3 restaurant_sim_V2.py for simple stats output
#python3 restaurant_sim_V2.py --detailed for an interval-level output (like the original)

from collections import deque
from dataclasses import dataclass, field
from math import ceil
import random
from typing import Optional

# SETTINGS

STEP = 5
OPEN_MINUTE = 10 * 60
CLOSE_MINUTE = 22 * 60
CAPACITY = 100
ARRIVAL_TRIALS = 6

SLOTS_PER_COOK = 3
TOTAL_KITCHEN_SLOTS = 4 * SLOTS_PER_COOK

# More active/queued orders increase preparation time.
KITCHEN_LOAD_WEIGHT = 0.50
STATION_LOAD_WEIGHT = 0.20

# Salad and Nachos get discounted.
QUICK_PREP_MINUTES = 15
DISCOUNTS = (0.00, 0.05, 0.10, 0.15)
PRICE_SENSITIVITY = 1.0

SATISFACTION_BASELINE = 4.0
SATISFACTION_PENALTY = 8.0
RECENT_WINDOW = 20

SEAT_PATIENCE_STEPS = (2, 8)   # 10-40 minutes
FOOD_PATIENCE_STEPS = (6, 15)  # 30-75 minutes
REPEAT_ARRIVAL_RATE = 0.25
MAX_CUSTOMER_PROFILES = 5000

DEMAND_VARIATION = (0.85, 1.15)

# Arrival probability at each hour. Values between points are interpolated.
# There are separate lunch and dinner peaks.
DEMAND_PROFILE = [
    (10 * 60, 0.15), (11 * 60, 0.34), (12 * 60, 0.58),
    (13 * 60, 0.48), (14 * 60, 0.28), (15 * 60, 0.18),
    (16 * 60, 0.23), (17 * 60, 0.43), (18 * 60, 0.65),
    (19 * 60, 0.65), (20 * 60, 0.49), (21 * 60, 0.28),
    (22 * 60, 0.00),
]


def clamp(value, low, high):
    return max(low, min(high, value))


def round_to_step(minutes):
    """Round preparation time up to the next five-minute interval."""
    return max(STEP, ceil(minutes / STEP) * STEP)

# MENU AND PRICING

@dataclass
class Dish:
    name: str
    ingredient: str
    food_cost: float
    regular_price: float
    prep_minutes: int
    minimum_price: float


class Menu:
    def __init__(self):
        self.dishes = [ #name, ingredient, ingredient cost, price, prep time, minimum price
            Dish("Steak", "Beef", 17.00, 39.00, 30, 29.25),
            Dish("Salad", "Greens", 4.00, 14.00, 10, 10.50),
            Dish("Nachos", "Cheese", 5.00, 16.00, 15, 12.00),
            Dish("Pasta", "Pasta", 6.50, 19.00, 20, 14.25),
        ]
        self.prices = {dish.name: dish.regular_price for dish in self.dishes}
        self.quick_dishes = {
            dish.name for dish in self.dishes
            if dish.prep_minutes <= QUICK_PREP_MINUTES
        }
        self.discount_level = 0

    def price(self, dish):
        return self.prices[dish.name]

    def set_discount(self, action):
        """Set one of four discount levels; action 0 restores normal prices."""
        if isinstance(action, bool) or action not in range(len(DISCOUNTS)):
            raise ValueError("Action must be an integer from 0 to 3.")

        discount = DISCOUNTS[action]
        for dish in self.dishes:
            target = (
                dish.regular_price * (1 - discount)
                if dish.name in self.quick_dishes
                else dish.regular_price
            )
            self.prices[dish.name] = round(
                clamp(target, dish.minimum_price, dish.regular_price), 2
            )
        self.discount_level = action

    def choose_dish(self, preferred_food, rng):
        """Customers favor their preferred dish and relatively cheaper items."""
        weights = []
        for dish in self.dishes:
            preference = 3.0 if dish.name == preferred_food else 1.0
            price_factor = (dish.regular_price / self.price(dish)) ** PRICE_SENSITIVITY
            weights.append(preference * price_factor)
        return rng.choices(self.dishes, weights=weights, k=1)[0]

# CUSTOMER PROFILES AND VISITS

@dataclass
class CustomerProfile:
    customer_id: int
    preferred_food: str
    seat_patience: int
    food_patience: int
    visit_count: int = 1
    last_visit_day: int = -1
    satisfaction_history: deque = field(default_factory=lambda: deque(maxlen=10))

    @property
    def average_satisfaction(self):
        return (
            sum(self.satisfaction_history) / len(self.satisfaction_history)
            if self.satisfaction_history else 3.5
        )


class CustomerRegistry:
    """Keeps customer preferences and satisfaction history across days."""

    def __init__(self):
        self.profiles = {}
        self.day = -1
        self.next_id = 1
        self.eligible_today = []

    def start_day(self):
        self.day += 1
        self.eligible_today = [
            customer_id for customer_id, profile in self.profiles.items()
            if profile.last_visit_day < self.day
        ]

    def _new_profile(self, rng):
        # Keep the profile store bounded for long training runs.
        if len(self.profiles) >= MAX_CUSTOMER_PROFILES:
            oldest_id = next(iter(self.profiles))
            del self.profiles[oldest_id]
            if oldest_id in self.eligible_today:
                self.eligible_today.remove(oldest_id)

        profile = CustomerProfile(
            customer_id=self.next_id,
            preferred_food=rng.choice(["Steak", "Salad", "Nachos", "Pasta"]),
            seat_patience=rng.randint(*SEAT_PATIENCE_STEPS) * STEP,
            food_patience=rng.randint(*FOOD_PATIENCE_STEPS) * STEP,
            last_visit_day=self.day,
        )
        self.profiles[self.next_id] = profile
        self.next_id += 1
        return profile, False

    def create_visit(self, rng):
        """Return a new customer or a previous customer who chooses to return."""
        if self.eligible_today and rng.random() < REPEAT_ARRIVAL_RATE:
            customer_id = rng.choice(self.eligible_today)
            self.eligible_today.remove(customer_id)
            profile = self.profiles[customer_id]
            return_probability = clamp(profile.average_satisfaction / 5, 0.05, 1.0)
            if rng.random() < return_probability:
                profile.visit_count += 1
                profile.last_visit_day = self.day
                return profile, True
        return self._new_profile(rng)

    def record_satisfaction(self, customer_id, rating):
        if customer_id in self.profiles:
            self.profiles[customer_id].satisfaction_history.append(rating)


@dataclass
class Customer:
    customer_id: int
    preferred_food: str
    seat_patience: int
    food_patience: int
    arrival_minute: int
    visit_number: int
    returning: bool

    status: str = "waiting_for_seat"
    seat_wait: int = 0
    order_minute: int = 0
    dish: Optional[Dish] = None
    price_paid: float = 0.0
    kitchen_order_id: Optional[int] = None
    food_wait: int = 0
    overall_rating: float = 0.0
    price_rating: float = 0.0
    wait_rating: float = 0.0

# KITCHEN

@dataclass
class KitchenOrder:
    order_id: int
    customer_id: int
    dish: Dish
    submitted: int
    ready: int = None
    abandoned: bool = False


@dataclass
class KitchenStation:
    active: list = field(default_factory=list)
    queue: deque = field(default_factory=deque)


class Kitchen:
    def __init__(self, menu):
        # Each menu item has its own cook, three active slots, and FIFO queue.
        self.stations = {dish.name: KitchenStation() for dish in menu.dishes}
        self.dishes = {dish.name: dish for dish in menu.dishes}
        self.next_order_id = 1

    @property
    def active_count(self):
        return sum(len(station.active) for station in self.stations.values())

    @property
    def queued_count(self):
        return sum(len(station.queue) for station in self.stations.values())

    @property
    def has_orders(self):
        return self.active_count + self.queued_count > 0

    def submit(self, customer, minute):
        order = KitchenOrder(
            order_id=self.next_order_id,
            customer_id=customer.customer_id,
            dish=customer.dish,
            submitted=minute,
        )
        self.next_order_id += 1
        self.stations[customer.dish.name].queue.append(order)
        return order

    def complete_ready(self, minute):
        """Remove finished orders and return them for delivery to customers."""
        completed = []
        for station in self.stations.values():
            still_active = []
            for order in station.active:
                if order.ready <= minute:
                    completed.append(order)
                else:
                    still_active.append(order)
            station.active = still_active
        return completed

    def start_waiting(self, minute):
        """Start queued orders whenever a cook has an available slot."""
        started = []
        for station in self.stations.values():
            while station.queue and len(station.active) < SLOTS_PER_COOK:
                total_pending = self.active_count + self.queued_count
                overall_load = clamp(total_pending / TOTAL_KITCHEN_SLOTS, 0, 1)
                station_load = len(station.active) / SLOTS_PER_COOK
                slowdown = 1 + KITCHEN_LOAD_WEIGHT * overall_load
                slowdown += STATION_LOAD_WEIGHT * station_load

                order = station.queue.popleft()
                prep_time = round_to_step(order.dish.prep_minutes * slowdown)
                order.ready = minute + prep_time
                station.active.append(order)
                started.append((order, prep_time))
        return started

    def abandon(self, order_id):
        """Remove a queued order or mark an active order as wasted."""
        for station in self.stations.values():
            for order in list(station.queue):
                if order.order_id == order_id:
                    station.queue.remove(order)
                    return "queued"
            for order in station.active:
                if order.order_id == order_id:
                    order.abandoned = True
                    return "active"
        return "not_found"

    def report(self):
        return {
            name: {"active": len(station.active), "queued": len(station.queue)}
            for name, station in self.stations.items()
        }

    def estimate_food_waits(self, minute):
        """Estimate how long it would take to make a dish at this working station at a particular time"""
        estimates = {}

        #Include another (hypothetical) order in the workflow
        pending = self.active_count + self.queued_count + 1
        overall_load = clamp(
            pending / TOTAL_KITCHEN_SLOTS,
            0,
            1
        )
        for name, station in self.stations.items():
            slot_ready = [
                max(minute, order.ready)
                for order in station.active
            ]
            slot_ready.extend(
                [minute] * (SLOTS_PER_COOK - len(slot_ready))
            )
            orders = [order.dish for order in station.queue]
            orders.append(self.dishes[name])
            candidate_ready = minute
            for dish in orders:
                slot = min(
                    range(SLOTS_PER_COOK),
                    key=lambda i: slot_ready[i]
                )
                start_time = slot_ready[slot]
                other_busy = sum(
                    1
                    for i in range(SLOTS_PER_COOK)
                    if i != slot and slot_ready[i] > start_time
                )
                station_load = other_busy / SLOTS_PER_COOK
                slowdown = (
                    1
                    + KITCHEN_LOAD_WEIGHT * overall_load
                    + STATION_LOAD_WEIGHT * station_load
                )
                prep_time = round_to_step(
                    dish.prep_minutes * slowdown
                )
                candidate_ready = start_time + prep_time
                slot_ready[slot] = candidate_ready

             # Total time from placing a new order until food is ready.
            estimates[name] = candidate_ready - minute

        return estimates

# RESTAURANT METRICS

@dataclass
class Stats:
    arrived: int = 0
    served: int = 0
    left_seat: int = 0
    left_food: int = 0
    new_visits: int = 0
    returning_visits: int = 0

    revenue: float = 0.0
    food_cost: float = 0.0
    max_occupied: int = 0
    max_seat_queue: int = 0
    max_kitchen_queue: int = 0
    max_kitchen_active: int = 0
    wasted_orders: int = 0

    food_waits: list = field(default_factory=list)
    customer_waits: deque = field(default_factory=lambda: deque(maxlen=RECENT_WINDOW))
    ratings: deque = field(default_factory=lambda: deque(maxlen=RECENT_WINDOW))
    feedback: list = field(default_factory=list)
    total_wait: float = 0.0
    satisfaction_total: float = 0.0
    satisfaction_count: int = 0

    @property
    def profit(self):
        return self.revenue - self.food_cost

    @property
    def average_food_wait(self):
        return sum(self.food_waits) / len(self.food_waits) if self.food_waits else 0.0

    @property
    def average_customer_wait(self):
        return self.total_wait / len(self.feedback) if self.feedback else 0.0

    @property
    def recent_wait(self):
        return sum(self.customer_waits) / len(self.customer_waits) if self.customer_waits else 0.0

    @property
    def average_satisfaction(self):
        return self.satisfaction_total / self.satisfaction_count if self.satisfaction_count else 0.0

    @property
    def recent_satisfaction(self):
        return sum(self.ratings) / len(self.ratings) if self.ratings else 0.0

    def record_feedback(self, customer, reason):
        if reason == "left_seat":
            total_wait = customer.seat_wait
        else:
            total_wait = customer.seat_wait + customer.food_wait

        self.customer_waits.append(total_wait)
        self.total_wait += total_wait
        self.ratings.append(customer.overall_rating)
        self.satisfaction_total += customer.overall_rating
        self.satisfaction_count += 1

        self.feedback.append({
            "customer_id": customer.customer_id,
            "visit_number": customer.visit_number,
            "returning_customer": customer.returning,
            "dish": customer.dish.name if customer.dish else None,
            "price_paid": customer.price_paid if customer.dish else None,
            "customer_wait_minutes": total_wait,
            "food_wait_minutes": customer.food_wait,
            "overall_satisfaction": customer.overall_rating,
            "price_satisfaction": customer.price_rating,
            "wait_satisfaction": customer.wait_rating,
            "exit_reason": reason,
        })

# RESTAURANT SIMULATION

class Restaurant:
    def __init__(self, seed=42, registry=None, vary_demand=True):
        self.rng = random.Random(seed)
        self.menu = Menu()
        self.kitchen = Kitchen(self.menu)
        self.registry = registry or CustomerRegistry()
        self.registry.start_day()
        self.stats = Stats()
        self.waiting = deque()
        self.seated = []
        self.customers = {}
        self.demand_points = self._make_demand_points(vary_demand)
        self.day = self.registry.day

    def _make_demand_points(self, vary):
        points = []
        for minute, base in DEMAND_PROFILE:
            factor = self.rng.uniform(*DEMAND_VARIATION) if vary and base else 1.0
            points.append((minute, clamp(base * factor, 0, 0.95)))
        return points

    def demand(self, minute):
        if minute < OPEN_MINUTE or minute >= CLOSE_MINUTE:
            return 0.0
        for (start, start_rate), (end, end_rate) in zip(
            self.demand_points, self.demand_points[1:]
        ):
            if start <= minute <= end:
                fraction = (minute - start) / (end - start)
                return start_rate + fraction * (end_rate - start_rate)
        return 0.0

    def add_arrivals(self, minute):
        count = sum(self.rng.random() < self.demand(minute) for _ in range(ARRIVAL_TRIALS))
        for _ in range(count):
            profile, returning = self.registry.create_visit(self.rng)
            customer = Customer(
                customer_id=profile.customer_id,
                preferred_food=profile.preferred_food,
                seat_patience=profile.seat_patience,
                food_patience=profile.food_patience,
                arrival_minute=minute,
                visit_number=profile.visit_count,
                returning=returning,
            )
            self.waiting.append(customer)
            self.stats.returning_visits += int(returning)
            self.stats.new_visits += int(not returning)
        self.stats.arrived += count
        return count

    def seat_customers(self, minute):
        while self.waiting and len(self.seated) < CAPACITY:
            customer = self.waiting.popleft()
            customer.status = "deciding"
            customer.seat_wait = minute - customer.arrival_minute
            # Customers take 0, 5, 10, or 15 minutes to order.
            delay = self.rng.choices([0, 5, 10, 15], [1, 6, 3, 1])[0]
            customer.order_minute = minute + delay
            self.seated.append(customer)
            self.customers[customer.customer_id] = customer

    def _rate_customer(self, customer, reason):
        """Generate transparent simulated 1-5 price, wait, and overall ratings."""
        if customer.dish:
            discount = clamp(
                (customer.dish.regular_price - customer.price_paid)
                / customer.dish.regular_price, 0, 1
            )
            price_rating = clamp(3.5 + 10 * discount, 1, 5)
        else:
            price_rating = 3.5

        if reason == "left_seat":
            total_wait = customer.seat_wait
            tolerance = customer.seat_patience
        else:
            total_wait = customer.seat_wait + customer.food_wait
            tolerance = customer.seat_patience + customer.food_patience

        wait_rating = clamp(5 - 3 * total_wait / max(1, tolerance), 1, 5)
        if reason in ("left_seat", "left_food"):
            wait_rating = min(wait_rating, 1.5)

        overall = clamp(
            0.65 * wait_rating + 0.35 * price_rating
            + self.rng.uniform(-0.15, 0.15),
            1, 5
        )
        customer.price_rating = round(price_rating, 2)
        customer.wait_rating = round(wait_rating, 2)
        customer.overall_rating = round(overall, 2)
        self.stats.record_feedback(customer, reason)
        self.registry.record_satisfaction(customer.customer_id, customer.overall_rating)

    def _finish_kitchen_orders(self, minute):
        for order in self.kitchen.complete_ready(minute):
            customer = self.customers.get(order.customer_id)
            if customer is None or customer.status != "waiting_for_food":
                continue

            customer.food_wait = order.ready - customer.order_minute
            self.stats.food_waits.append(customer.food_wait)
            customer.status = "eating"
            meal_time = self.rng.choices(
                list(range(30, 91, 5)),
                [max(1, 13 - abs(m - 45) / 5) for m in range(30, 91, 5)],
            )[0]
            customer.finish_minute = minute + meal_time

    def _leave(self, customer, reason):
        if reason == "left_food":
            result = self.kitchen.abandon(customer.kitchen_order_id)
            customer.food_wait = max(0, self.current_minute - customer.order_minute)
            if result == "active":
                self.stats.wasted_orders += 1

        if reason == "left_seat":
            self.stats.left_seat += 1
        elif reason == "left_food":
            self.stats.left_food += 1
        elif reason == "served":
            self.stats.served += 1
            self.stats.revenue += customer.price_paid

        if reason == "left_food":
            self.stats.food_waits.append(customer.food_wait)

        self._rate_customer(customer, reason)
        self.customers.pop(customer.customer_id, None)

    def _process_customers(self, minute):
        self.current_minute = minute
        remaining = []

        for customer in self.seated:
            if customer.status == "deciding":
                if minute < customer.order_minute:
                    remaining.append(customer)
                    continue

                customer.dish = self.menu.choose_dish(customer.preferred_food, self.rng)
                customer.price_paid = self.menu.price(customer.dish)
                customer.order_minute = minute
                customer.status = "waiting_for_food"
                order = self.kitchen.submit(customer, minute)
                customer.kitchen_order_id = order.order_id
                remaining.append(customer)

            elif customer.status == "waiting_for_food":
                if minute - customer.order_minute >= customer.food_patience:
                    self._leave(customer, "left_food")
                else:
                    remaining.append(customer)

            elif customer.status == "eating":
                if minute >= customer.finish_minute:
                    self._leave(customer, "served")
                else:
                    remaining.append(customer)

        self.seated = remaining

    def _process_seat_queue(self, minute):
        remaining = deque()
        for customer in self.waiting:
            wait = minute - customer.arrival_minute
            if wait >= customer.seat_patience:
                customer.seat_wait = wait
                self._rate_customer(customer, "left_seat")
                self.stats.left_seat += 1
            else:
                remaining.append(customer)
        self.waiting = remaining

    def step(self, minute):
        # Finish orders first, so a meal ready at this tick reaches its customer before the customer's patience check.
        self._finish_kitchen_orders(minute)
        self._process_customers(minute)
        self._process_seat_queue(minute)
        arrivals = self.add_arrivals(minute) if minute < CLOSE_MINUTE else 0
        self.seat_customers(minute)

        # Cancelled queued orders are removed before this method starts new work.
        started = self.kitchen.start_waiting(minute)
        for order, _prep_time in started:
            self.stats.food_cost += order.dish.food_cost

        self.stats.max_occupied = max(self.stats.max_occupied, len(self.seated))
        self.stats.max_seat_queue = max(self.stats.max_seat_queue, len(self.waiting))
        self.stats.max_kitchen_queue = max(
            self.stats.max_kitchen_queue, self.kitchen.queued_count
        )
        self.stats.max_kitchen_active = max(
            self.stats.max_kitchen_active, self.kitchen.active_count
        )
        return arrivals

    def run(self, detailed=False):
        """Run a 12-hour arrival day and then finish all remaining customers."""
        minute = OPEN_MINUTE
        while (
            minute < CLOSE_MINUTE or self.waiting or self.seated or self.kitchen.has_orders
        ):
            new_customers = self.step(minute)
            if detailed:
                print(
                    f"{minute // 60:02d}:{minute % 60:02d} | "
                    f"New: {new_customers} | Seated: {len(self.seated)} | "
                    f"Seat queue: {len(self.waiting)} | "
                    f"Kitchen active: {self.kitchen.active_count} | "
                    f"Kitchen queue: {self.kitchen.queued_count} | "
                    f"Served: {self.stats.served} | "
                    f"Left: {self.stats.left_seat + self.stats.left_food}"
                )
            minute += STEP
        return self.stats

# RL-READY ENVIRONMENT INTERFACE (NO RL MODEL YET... I just wanted to get the sim running)

class RestaurantEnv:
    """One 12-hour simulated day with observations, actions, and reward.

    Actions: 0=no discount, 1=5%, 2=10%, 3=15% on Salad and Nachos only.
    Reward: period profit minus a penalty when new customer ratings average
    below the 4/5 satisfaction baseline. No learning algorithm is included.
    """

    OBSERVATION_LABELS = (
        "time", "seat_queue", "occupied_seats", "recent_wait", "recent_satisfaction",
        "demand", "deciding", "waiting_for_food", "eating", "kitchen_active",
        "kitchen_queue", "returning_share", "discount_level",
        "steak_active", "salad_active", "nachos_active", "pasta_active",
        "steak_queue", "salad_queue", "nachos_queue", "pasta_queue",
        "steak_estimated_wait", "salad_estimated_wait", "nachos_estimated_wait",
        "pasta_estimated_wait",
    )

    def __init__(self, seed=42, registry=None, vary_demand=True):
        self.seed = seed
        self.registry = registry or CustomerRegistry()
        self.vary_demand = vary_demand
        self.reset()

    @property
    def observation_size(self):
        return len(self.OBSERVATION_LABELS)

    @property
    def action_count(self):
        return len(DISCOUNTS)

    def reset(self, seed=None):
        if seed is not None:
            self.seed = seed
        self.restaurant = Restaurant(self.seed, self.registry, self.vary_demand)
        self.seed += 1
        self.minute = OPEN_MINUTE
        self.done = False
        return self.observation(), {
            "day": self.restaurant.day,
            "prices": self.restaurant.menu.prices.copy(),
            "quick_dishes": sorted(self.restaurant.menu.quick_dishes),
            "discount_levels": DISCOUNTS,
        }

    def observation(self):
        r = self.restaurant
        stats = r.stats
        statuses = {name: 0 for name in ("deciding", "waiting_for_food", "eating")}
        for customer in r.seated:
            if customer.status in statuses:
                statuses[customer.status] += 1

        station = r.kitchen.report()
        estimated_waits = r.kitchen.estimate_food_waits(self.minute)
        values = [
            clamp((self.minute - OPEN_MINUTE) / (CLOSE_MINUTE - OPEN_MINUTE), 0, 1),
            len(r.waiting) / CAPACITY,
            len(r.seated) / CAPACITY,
            clamp(stats.recent_wait / 120, 0, 1.5),
            stats.recent_satisfaction / 5 if stats.ratings else SATISFACTION_BASELINE / 5,
            r.demand(self.minute),
            statuses["deciding"] / CAPACITY,
            statuses["waiting_for_food"] / CAPACITY,
            statuses["eating"] / CAPACITY,
            r.kitchen.active_count / TOTAL_KITCHEN_SLOTS,
            r.kitchen.queued_count / (TOTAL_KITCHEN_SLOTS * 3),
            stats.returning_visits / stats.arrived if stats.arrived else 0.0,
            r.menu.discount_level / (len(DISCOUNTS) - 1),
        ]
        values.extend(station[name]["active"] / SLOTS_PER_COOK
                      for name in ("Steak", "Salad", "Nachos", "Pasta"))
        values.extend(station[name]["queued"] / (SLOTS_PER_COOK * 3)
                      for name in ("Steak", "Salad", "Nachos", "Pasta"))
        values.extend(clamp(estimated_waits[name] / 120, 0, 2)
                      for name in ("Steak", "Salad", "Nachos", "Pasta"))
        return tuple(float(value) for value in values)

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode ended. Call reset() before stepping again.")

        if isinstance(action, bool) or action not in range(len(DISCOUNTS)):
            raise ValueError("Action must be an integer from 0 to 3.")

        r = self.restaurant
        old_profit = r.stats.profit
        old_feedback_count = len(r.stats.feedback)

        # The agent controls prices every five minutes during opening hours.
        if self.minute < CLOSE_MINUTE:
            r.menu.set_discount(action)

        r.step(self.minute)
        self.minute += STEP

        # After closing, finish existing customers and orders without asking the agent for extra decisions. This keeps one episode to one day.
        if self.minute >= CLOSE_MINUTE:
            while r.waiting or r.seated or r.kitchen.has_orders:
                r.step(self.minute)
                self.minute += STEP
            self.done = True

        new_feedback = r.stats.feedback[old_feedback_count:]
        if new_feedback:
            average_rating = sum(
                item["overall_satisfaction"] for item in new_feedback
            ) / len(new_feedback)
            penalty = (
                SATISFACTION_PENALTY
                * max(0.0, SATISFACTION_BASELINE - average_rating)
                * len(new_feedback)
            )
        else:
            average_rating = None
            penalty = 0.0

        period_profit = r.stats.profit - old_profit
        reward = period_profit - penalty

        info = {
            "day": r.day,
            "action": action,
            "discount": DISCOUNTS[action],
            "prices": r.menu.prices.copy(),
            "period_profit": period_profit,
            "satisfaction_penalty": penalty,
            "period_average_satisfaction": average_rating,
            "reward": reward,
            "net_profit": r.stats.profit,
            "served": r.stats.served,
            "average_customer_wait": r.stats.average_customer_wait,
            "average_food_wait": r.stats.average_food_wait,
            "average_satisfaction": r.stats.average_satisfaction,
            "left_seat": r.stats.left_seat,
            "left_food": r.stats.left_food,
            "returning_visits": r.stats.returning_visits,
            "kitchen": r.kitchen.report(),
            "estimated_food_wait_minutes": r.kitchen.estimate_food_waits(self.minute),
            "new_feedback": new_feedback, 
        }
        return self.observation(), reward, self.done, False, info

# BASIC CHECKS AND STANDALONE SIMULATION

def run_smoke_tests():
    """Check station capacity, FIFO queues, pricing limits, and RL interface."""
    menu = Menu()
    kitchen = Kitchen(menu)
    steak = next(dish for dish in menu.dishes if dish.name == "Steak")

    # Four steak orders: three start and the fourth waits.
    customers = [
        Customer(i, "Steak", 20, 60, OPEN_MINUTE, 1, False, dish=steak)
        for i in range(1, 5)
    ]
    for customer in customers:
        kitchen.submit(customer, OPEN_MINUTE)
    kitchen.start_waiting(OPEN_MINUTE)
    assert len(kitchen.stations["Steak"].active) == 3
    assert len(kitchen.stations["Steak"].queue) == 1
    first_ready = min(order.ready for order in kitchen.stations["Steak"].active)
    kitchen.complete_ready(first_ready)
    started = kitchen.start_waiting(first_ready)
    assert len(started) == 1 and started[0][0].order_id == 4

    # Each of the four cooks independently has three active slots.
    all_kitchen = Kitchen(menu)
    order_id = 10
    for dish in menu.dishes:
        for _ in range(4):
            customer = Customer(
                order_id, dish.name, 20, 60, OPEN_MINUTE, 1, False, dish=dish
            )
            all_kitchen.submit(customer, OPEN_MINUTE)
            order_id += 1
    all_kitchen.start_waiting(OPEN_MINUTE)
    for station in all_kitchen.stations.values():
        assert len(station.active) == 3 and len(station.queue) == 1

    # Prices only discount quick dishes, respect floors, and restore at action 0.
    menu.set_discount(3)
    for dish in menu.dishes:
        assert dish.minimum_price <= menu.price(dish) <= dish.regular_price
    assert menu.price(steak) == steak.regular_price
    assert menu.price(next(d for d in menu.dishes if d.name == "Salad")) < 14.00
    menu.set_discount(0)
    assert all(menu.price(d) == d.regular_price for d in menu.dishes)

    env = RestaurantEnv(seed=100, vary_demand=False)
    state, _, terminated, truncated, info = env.step(2)
    assert len(state) == env.observation_size == 25
    estimated_waits = info["estimated_food_wait_minutes"]
    assert len(estimated_waits) == 4
    assert all(
        wait >= 0
        for wait in estimated_waits.values()
    )
    assert env.action_count == 4
    assert not terminated and not truncated
    assert info["prices"]["Steak"] == 39.00
    assert info["prices"]["Salad"] < 14.00


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run the restaurant simulation only.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--detailed", action="store_true")
    parser.add_argument("--skip-checks", action="store_true")
    args = parser.parse_args()

    if not args.skip_checks:
        run_smoke_tests()
        print("Simulation checks passed.")

    restaurant = Restaurant(seed=args.seed)
    stats = restaurant.run(detailed=args.detailed)

    print("\nRESTAURANT SIMULATION RESULTS")
    print(f"Arrivals: {stats.arrived}")
    print(f"New visits: {stats.new_visits}")
    print(f"Returning visits: {stats.returning_visits}")
    print(f"Served: {stats.served}")
    print(f"Left waiting for seats: {stats.left_seat}")
    print(f"Left waiting for food: {stats.left_food}")
    print(f"Average customer wait: {stats.average_customer_wait:.1f} minutes")
    print(f"Average food wait: {stats.average_food_wait:.1f} minutes")
    print(f"Average satisfaction: {stats.average_satisfaction:.2f}/5")
    print(f"Revenue: ${stats.revenue:.2f}")
    print(f"Food cost: ${stats.food_cost:.2f}")
    print(f"Net profit: ${stats.profit:.2f}")
    print("\nKITCHEN STATIONS")
    for name, data in restaurant.kitchen.report().items():
        print(f"{name}: {data['active']}/{SLOTS_PER_COOK} active, {data['queued']} queued")


if __name__ == "__main__":
    main()
