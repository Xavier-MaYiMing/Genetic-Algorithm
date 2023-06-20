### Genetic Algorithm

##### Reference: Holland J H. Genetic algorithms[J]. Scientific American, 1992, 267(1): 66-73.

The genetic algorithm (GA) with simulated binary crossover and polynomial mutation to solve real optimization problems.

| Variables | Meaning                               |
| --------- | ------------------------------------- |
| npop      | population size                       |
| iter      | iteration number                      |
| lb        | lower bound                           |
| ub        | upper bound                           |
| pc        | crossover probability                 |
| eta_c     | Spread factor distribution index      |
| pm        | mutation probability                  |
| eta_m     | perturbance factor distribution index |
| dim       | dimension                             |
| pop       | population                            |
| objs      | objectives                            |
| gbest     | the global best                       |
| gbest_sol | the global best solution              |
| iter_best | the global best of each iteration     |
| con_iter  | convergence iteration                 |

#### Test problem: Pressure vessel design

![](https://github.com/Xavier-MaYiMing/Genetic-Algorithm/blob/main/Pressure%20vessel%20design.png)

$$
\begin{align}
&\text{min}\ f(x)=0.6224x_1x_3x_4+1.7781x_2x_3^2+3.1661x_1^2x_4+19.84x_1^2x_3,\\
&\text{s.t.} \\
&-x_1+0.0193x_3\leq0,\\
&-x_3+0.0095x_3\leq0,\\
&-\pi x_3^2x_4-\frac{4}{3}\pi x_3^3+1296000\leq0,\\
&x_4-240\leq0,\\
&0\leq x_1\leq99,\\
&0\leq x_2 \leq99,\\
&10\leq x_3 \leq 200,\\
&10\leq x_4 \leq 200.
\end{align}
$$

#### Example

```python
if __name__ == '__main__':
    t_npop = 300
    t_iter = 1500
    t_lb = np.array([0, 0, 10, 10])
    t_ub = np.array([99, 99, 200, 200])
    print(main(t_npop, t_iter, t_lb, t_ub))
```

##### Output:

![](https://github.com/Xavier-MaYiMing/Genetic-Algorithm/blob/main/convergence%20curve.png)

The GA converges at its 1,331-th iteration, and the global best value is 8051.400862765215. 

```python
{
  'gbest': 8051.400862765215, 
  'best solution': array([ 1.3005842 ,  0.64288936, 67.38671732, 10.        ]), 
  'convergence iteration': 1331
}
```

