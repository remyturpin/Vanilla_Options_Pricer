##  Vanilla Options Pricer

#  Features
This Vanilla Options Pricer is a good tool to compare monte carlo and black and scholes option prices with some graphs.
It is realized with Flask and Python to integrate HTML transforming it in a web app, the design is realized in CSS.

- **Compares option pricing models**: Black-Scholes vs. Monte Carlo simulation.
- **Supports European call and put options**.
- **Computes option Greeks**: Delta, Gamma, Vega, Theta, and Rho.
- **Generates interactive plots**:
  - Option price evolution over time.
  - Option price and Greeks vs. underlying asset price.
- **Built as a web application using Flask** with a user-friendly interface.
- **Styled with CSS** for enhanced UI design.

# Code and libraries used
- **Flask**: Web framework for building the API and front-end interface.
- **NumPy**: Used for numerical computations, including Monte Carlo simulations.
- **SciPy**: Utilized for statistical functions such as cumulative distribution in Black-Scholes.
- **Matplotlib**: Generates visualizations for option price trends and Greek sensitivities.
- **HTML & CSS**: Provides a simple and interactive front-end for users.
