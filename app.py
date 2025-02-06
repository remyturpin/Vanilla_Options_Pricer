from flask import Flask, render_template, request, jsonify
import numpy as np
import scipy.stats as si
import matplotlib.pyplot as plt
import io
import base64

def black_scholes(S, K, T, r, sigma, option_type="call"):
    """Calcule le prix d'une option européenne avec Black-Scholes."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == "call":
        price = S * si.norm.cdf(d1) - K * np.exp(-r * T) * si.norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * si.norm.cdf(-d2) - S * si.norm.cdf(-d1)
    
    return price

def monte_carlo_simulation(S, K, T, r, sigma, option_type, simulations=10000):
    """Monte Carlo simulation for option pricing."""
    np.random.seed(42)
    ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * np.random.randn(simulations))
    
    if option_type == "call":
        payoff = np.maximum(ST - K, 0)
    else:
        payoff = np.maximum(K - ST, 0)
    
    return np.exp(-r * T) * np.mean(payoff)

def greeks(S, K, T, r, sigma, option_type="call"):
    """Calcule les grecs Delta, Gamma, Vega, Theta, Rho."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    delta = si.norm.cdf(d1) if option_type == "call" else si.norm.cdf(d1) - 1
    gamma = si.norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * si.norm.pdf(d1) * np.sqrt(T)
    theta = (-S * si.norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * si.norm.cdf(d2))
    rho = K * T * np.exp(-r * T) * si.norm.cdf(d2) if option_type == "call" else -K * T * np.exp(-r * T) * si.norm.cdf(-d2)
    
    return {
        "Delta": delta,
        "Gamma": gamma,
        "Vega": vega / 100,
        "Theta": theta / 365,
        "Rho": rho / 100
    }

def generate_plots(S, K, T, r, sigma, option_type):
    """Génère deux graphiques distincts : option price et grecs vs temps et vs prix du sous-jacent."""
    T_range = np.linspace(0.01, T, 100)
    S_range = np.linspace(S * 0.5, S * 1.5, 100)
    
    prices_T = [black_scholes(S, K, t, r, sigma, option_type) for t in T_range]
    prices_S = [black_scholes(s, K, T, r, sigma, option_type) for s in S_range]
    
    greeks_T = {g: [greeks(S, K, t, r, sigma, option_type)[g] for t in T_range] for g in ["Delta", "Gamma", "Vega", "Theta", "Rho"]}
    greeks_S = {g: [greeks(s, K, T, r, sigma, option_type)[g] for s in S_range] for g in ["Delta", "Gamma", "Vega", "Theta", "Rho"]}
    
    # Générer le premier graphique (évolution avec le temps)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(T_range, prices_T, label='Option Price', linewidth=2)
    for g in greeks_T:
        ax.plot(T_range, greeks_T[g], label=g, linestyle='dashed')
    ax.set_xlabel('Temps avant expiration')
    ax.set_title('Évolution du prix et des grecs en fonction du temps')
    ax.legend()
    ax.grid(True)
    img1 = io.BytesIO()
    plt.savefig(img1, format='png')
    img1.seek(0)
    plot_url1 = base64.b64encode(img1.getvalue()).decode()
    plt.close()
    
    # Générer le deuxième graphique (évolution avec le prix du sous-jacent)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(S_range, prices_S, label='Option Price', linewidth=2)
    for g in greeks_S:
        ax.plot(S_range, greeks_S[g], label=g, linestyle='dashed')
    ax.set_xlabel('Prix du sous-jacent')
    ax.set_title('Évolution du prix et des grecs en fonction du sous-jacent')
    ax.legend()
    ax.grid(True)
    img2 = io.BytesIO()
    plt.savefig(img2, format='png')
    img2.seek(0)
    plot_url2 = base64.b64encode(img2.getvalue()).decode()
    plt.close()
    
    return plot_url1, plot_url2

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/price', methods=['POST'])
def price():
    data = request.json
    S = float(data['S'])
    K = float(data['K'])
    T = float(data['T'])
    r = float(data['r'])
    sigma = float(data['sigma'])
    option_type = data['option_type']
    
    bs_price = black_scholes(S, K, T, r, sigma, option_type)
    mc_price = monte_carlo_simulation(S, K, T, r, sigma, option_type)
    greeks_values = greeks(S, K, T, r, sigma, option_type)
    plot_url1, plot_url2 = generate_plots(S, K, T, r, sigma, option_type)
    
    return jsonify({"black_scholes": bs_price, "monte_carlo": mc_price, "greeks": greeks_values, "plot_url1": plot_url1, "plot_url2": plot_url2})

if __name__ == '__main__':
    app.run(debug=True)
