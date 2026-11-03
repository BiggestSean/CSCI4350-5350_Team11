# CSCI4350-5350_Team11
Here we will collaboratively work on and iterate on our CSCI Project.

Flexible Pricing to Optimize Customer Satisfaction 
CSCI 4350/5350 · Project Abstract · Group 11 

Team members: Bijay Khanal (CSCI 5350), Sean Reyes, Brandon Smith 

Option chosen: Reinforcement learning in a simulated restaurant 

1. The Problem 

Customers are the foundation of every restaurant, and a satisfied customer is far more likely to 
return and to recommend the restaurant to others. Long waits during peak hours are one of the 
most common causes of dissatisfaction: as the dining room fills, the kitchen slows down for every 
order, and each additional customer makes the experience worse for the people already there. 
Flexible (dynamic) pricing is usually used the other way around. It raises prices when demand is 
high, which can feel like price gouging when alternatives are limited and typically lowers 
satisfaction. 
Our project asks the opposite question: can an agent learn to lower the price of menu items that 
are quick to prepare during busy periods, so that customers voluntarily choose them? If enough 
customers shift toward fast items, the kitchen load drops, average waits shrink, seats turn over 
sooner, and more customers can be served in the same period. Prices stay profitable, but the 
incentive is aimed at improving the customer’s experience rather than extracting more from it. 

2. Data and Sources: A Simulated Restaurant 

No real customer data is used. A real restaurant would not let a learning agent experiment with 
prices on live customers, and an agent that learns from its own attempts needs thousands of days 
of experience, so we build a simulator. It has two parts that interact. 
Part A: the restaurant and its customers. Each cycle represents ~12 hours of operation. The 
restaurant has limited seating, a menu with a typical price and a minimum allowed price for each 
item, and a base preparation time for each item. Preparation times for all items increase as 
customer volume increases, which models a kitchen that gets overloaded. Customers arrive at 
varying rates through the day, with a lunch and dinner rush. Each customer has their own wait time 
tolerance and chooses an item stochastically, with price as one of the factors that determine the 
choice. A customer who waits longer than their tolerance, or who finds no seat, leaves unhappy or 
does not stay. After the meal, each customer leaves ratings for overall satisfaction, price and wait 
time. 
Part B: data sources and feedback. Menu items, prices, preparation times and arrival patterns 
will be modeled on publicly posted menus and typical service times from a small set of restaurants 
that we will select and cite in the final report. Because the customers are simulated, all feedback in 
this project is simulated, and we will state that clearly in the report. 

3. Reinforcement Learning Formulation 

The agent repeatedly acts (sets prices), receives feedback (customer ratings and business 
results), and changes its behavior to make fewer mistakes. We formulate this as follows. 
Group 11 - Flexible Pricing - Page 1 
In our formulation, the state is what the agent can observe about the restaurant at a given 
moment: the time of day, the queue length, the number of occupied seats, the recent average wait 
time and the recent average customer satisfaction. The action is the discount level the agent 
applies to the quick to prepare menu items for the next period (for example none, small, medium or 
large, never below the minimum price). The policy is the rule that maps each state to a discount 
level, which the agent learns from experience. The reward is the profit earned in the period, 
reduced by a penalty whenever average customer satisfaction falls below a set baseline. One 
episode is one simulated 12-hour day. 

4. Feedback, Reward Design and Reward Hacking

The agent receives feedback in the form of customer ratings on key metrics (satisfaction, price, 
wait time), the number of sales, repeat customers, and profit. Its reward is based on profit, with a 
baseline level of customer satisfaction acting as a guardrail. The guardrail matters because an 
agent will optimize exactly what it is rewarded for, including loopholes. We expect and will look for 
behaviors such as: 
• discounting every item all the time, which maximizes sales volume but destroys margins, or 
discounting so heavily that fast items crowd out the rest of the menu. 
• pushing customers away who want slower items, which can look like shorter waits while 
quietly hurting satisfaction and long-term repeat visits. 
• exploiting quirks of the simulator, such as how ratings are generated, instead of improving the 
real customer experience. 
We will inspect sampled simulated days for these behaviors. Any hack we find will be described in 
the report along with how we changed the reward to fix it. The report will also justify each reward 
term and its weight, explain what could go wrong with the reward, and describe how we checked. 

5. How the System Learns From Its Mistakes 

Training runs for many simulated days. On each day the agent sets discounts as the day unfolds, 
the simulated customers respond, and the day’s ratings, sales, repeat visits and profit are turned 
into rewards. The agent updates its policy so that discount choices that led to short waits, satisfied 
customers and healthy profit become more likely in similar situations, and choices that led to long 
queues, unhappy customers or losses become less likely. Early in training the agent explores a 
range of discount levels then later it relies more on what it has learned. We will keep a log of every 
day (states, actions, ratings, reward) so improvement can be traced and plotted. 

6. How We Will Measure Improvement 

Showing that the system improved from its own attempts, rather than just reporting a final number, 
is the central requirement. We will therefore: 
• Compared against baselines run in the same simulator: fixed prices with no discounts, 
random discounts, and a simple handwritten rule (for example, discount fast items whenever 
the queue exceeds a threshold). 
• Evaluate on held-out days, using new random seeds and demand patterns that the agent 
never trained on, so that results are not just memorization of the training days. 
Group 11 - Flexible Pricing - Page 2 
• Report the main metrics after every cycle: average customer wait time, net profit, average 
satisfaction, and number of customers served. 
• Plot learning curves of reward, wait time and profit over training, to show improvement 
happening over time. 
• Plot the trade-off between wait time and profit. Each policy (baselines and learned) is a 
point; our definition of success is a learned policy that reaches a point where wait times and 
profit are balanced, so that neither can be improved without making the other worse, and that 
beats the baselines on held-out days. 
• Analyze mistakes: show at least three situations (for example a lunch rush) where the early 
agent chose poorly and the trained agent chose better, with the before and after outcomes.

7. Limitations and Ethics 

Results from a simulator depend on how realistic the simulated customers are, and we cannot 
claim that the learned policy would work in a real restaurant without real world testing. We will test 
how sensitive the results are to the simulator’s assumptions (for example wait-time tolerance and 
price sensitivity). We also want the policy to be fair to customers: discounts should reduce prices 
and steer behavior, not mislead anyone, and we will not optimize profit at the expense of the 
satisfaction guardrail. 
