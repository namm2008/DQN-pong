# DQN-pong
Pong was a simulating game of table tennis. A ball was set to move and bounce in a box. Players control a pad to rebound the ball in order to avoid the ball to fall on their side. If the ball fell on the ones side, the opponent won a point. The winner would be the one who gained 21 points first.
The Pong environment was provided by OpenAI. The package called Gym provided the RGB screen which was an (210,160,3) array and the actions of the agent could be either UP or DOWN given that the computer player only moved Up and Down periodically. At the beginning of each point, the ball was set to start at the centre and moved towards the one who just lost. 

## Deep Q Learning (DQN)
In the Pong environment, there were 2 main problems of using Q learning. Firstly, the numerous different state-action combinations made it impossible to create the R-Matrix in advance. Secondly, the ineffective caused by the sparseness of the rewards given to the actions.

In Deep Q Learning, a neural network was used to approximate the Q value function. It was a combination of Deep Learning (DL) and Q-Learning. In each step, image of the screen (as the state) was inputted to a neural network and getting output as the probability of actions (Q-value). In this part, 2 different networks were compared.

## Customized DQN setting for Pong Environment
i. Preprocessing and state defining
The original screen pixel shape was (210,160,3). Some irrelevant pixels were cropped out. The remaining pixel shape was (160,160,3). The colour of the image was not important for deciding the actions, thus the screen was further decolour to form black and white single channel. The image was then resized to (80,80,1) which was shown in figure 12. For more efficient learning, the state was set to the difference of the current screen and last screen instead of concatenating of the screens. This could save half of the computation cost.

ii. Discounted rewards
The environment provided the rewards only when the player won or lost points. In between the game, zero rewards were assigned to every action. The sparse rewards allocation made the agent difficult to learn from the environment. A good strategy to allocate the rewards could improve the effective of training drastically. In order to encourage the agent to move like it won the point and discourage when it lost, positive rewards and negative rewards were allocated throughout the wining series and losing series of actions it took respectively. The actions which were closer to the winning or losing the points were more crucial. Thus, discounted rewards were assigned exponentially to the actions closed to the wining point and losing points. The Discounted factor was also a hyperparameters to be set before training.

To be more aggressive and improve the efficiency of training, the timing of allocating of rewards by the environment were also taken into account. The environment returned rewards when the ball surpassed the edge of the screen. However, when we consider the effective actions contributing to the winning or losing was not at those frames. To make more sense, the discounted rewards should be started to allocate to the actions that finally touch the pad of the agent which was 45 frames before the environment returned +1 reward, and the actions that finally not touching the pad which was 5 frames before it returned -1 reward. 

iii. Loss Functions and Updating scheme
Loss function: Mean Square Error

loss = MSE( ( R + γ max⁡(Q(s',a',ω)) ) , Q(s,a,ω) )

where R is the current reward, γ is the discount factor, s'is the next state a'is the next actions and ω is the parameters of the network. 
Update the policy network when the environment offered rewards after the discount rewards were allocated. This could reduce the training time by avoiding too frequent backward pass of the network and keeping the meaningful updates. Target Network was updated every 10 episodes. 

iv. Positive Reward Boosted Experience Replay
The states-action-reward experience was stored in the replay memory for updating the policy network. However, at the beginning of the training, the ratio of positive reward transitions to negative reward transitions was very small (about 2%). The randomly picked experience from the replay memory would not include enough positive experience which meant the policy was not updated efficiently. For this, the random sampling was replaced by separating positive and negative rewards transitions random sampling. At the beginning of the training, at least 20% of the positive rewards were drawn for each batch. The sampling positive ratio increased exponentially when the training went by. 

## Neural Network

Two different Network structures were tested:
1.	Linear Network
The input state preprocessed so that they simply contained zeros and ones. It was worth trying a simple network as the baseline model. One hidden layer network of 200 neurons with ReLu activation function was used.
2.	Convolutional Neural Network (CNN)
A 3 layers model of CNN was used. All the three layers were used with kernel of 5 and stride of 2. First, second and third layer used 16, 32 and 4 filters correspondingly. Batch Normalization was carried out in each layer. 
