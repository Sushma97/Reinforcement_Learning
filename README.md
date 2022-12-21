# Reinforcement_Learning

https://courses.grainger.illinois.edu/cs440/fa2022/MPs/mp6/assignment6.html

Snake is a famous video game originated in the 1976 arcade game Blockade. The player uses up, down, left and right to control the snake which grows in length (when it eats the food pellet), with the snake body and walls around the environment being the primary obstacle. In this assignment, we train an AI agent using reinforcement learning to play a simple version of the game snake. We implement a TD version of the Q-learning algorithm.

To see the available parameters you can set for the game, run:

```python mp6.py --help```

To train and test your agent, run:

```python mp6.py [parameters]```

For example, to run Test 1 above, run:

```python mp6.py --snake_head_x 5 --snake_head_y 5 --food_x 2 --food_y 2 --Ne 40 --C 40 --gamma 0.7```

## I hereby state that I shall not be held responsible for any misuse of my work or any academic integrity violations. ##
## DO NOT COPY. Only for reference ##