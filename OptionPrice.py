import jax
from jax import lax, numpy as jnp, grad, jit, random, scipy as jsp

import numpy as np
import scipy as sp
import pandas as pd

import yfinance as yf

from datetime import datetime, timedelta
import math
from typing import Union, Optional, Callable

class OptionPrice:

  def __init__(self, underlying_asset_current_price: float, strike_price: float,
               risk_free_rate: float, expiration_time: float, annual_vol: Optional[float] = None,
               target_stock: Optional[str] = None):
    self._s = underlying_asset_current_price
    self._k = strike_price
    self._r = risk_free_rate
    self._t = expiration_time
    if annual_vol:
      self._sigma = annual_vol
    else:
      if not target_stock:
        raise ValueError("Parameter `target_stock` expected dtype: str, got None")
      self._sigma = OptionPrice.__calc_annual_vol(OptionPrice.__get_hist_data(target_stock))
    
    self._call_greeks = None
    self._put_greeks = None
    
  

  def calc_call_price(self, s, sigma, k, t, r) -> float:
    b = jnp.exp(-r * t)

    x1 = (jnp.log(s / k) + (r + sigma ** 2 / 2) * t) / (sigma * jnp.sqrt(t))
    x2 = x1 - sigma * jnp.sqrt(t)

    self.call_price = s * jsp.stats.norm.cdf(x1) - jsp.stats.norm.cdf(x2) * k * b

    return self.call_price
  

  
  @property
  def call_greeks(self):
    return self._call_greeks
  
  @call_greeks.setter
  def call_greeks(self, func: Callable):
    self.grad_func = grad(func, (0, 1, 3, 4))
    self.call_delta, self.call_vega, self.call_theta, self.call_rho = self.grad_func(self._s, self._sigma, self._k, self._t, self._r)
    self.call_theta /= -365
    self.call_vega /= 100
    self.call_rho /= 100
    self._call_greeks = {'call_price': round(float(self.call_price.primal), 3), 'call_delta': round(float(self.call_delta), 3), 
    'call_vega': round(float(self.call_vega), 3), 'call_theta': round(float(self.call_theta), 3), 'call_rho': round(float(self.call_rho), 3)}
  

  def calc_put_price(self, s, sigma, k, t, r) -> float:
    b = jnp.exp(-r * t)

    x1 = (jnp.log(s / k) + (r + sigma ** 2 / 2) * t) / (sigma * jnp.sqrt(t))
    x2 = x1 - sigma * jnp.sqrt(t)

    self.put_price = k * b * jsp.stats.norm.cdf(-x2) - s * jsp.stats.norm.cdf(-x1)

    return self.put_price


  @property
  def put_greeks(self):
    return self._put_greeks
  
  @put_greeks.setter
  def put_greeks(self, func: Callable):
    self.grad_func = grad(func, (0, 1, 3, 4))
    self.put_delta, self.put_vega, self.put_theta, self.put_rho = self.grad_func(self._s, self._sigma, self._k, self._t, self._r)
    self.put_theta /= -365
    self.put_vega /= 100
    self.put_rho /= 100
    self._put_greeks = {'put_price': round(float(self.put_price.primal), 3), 'put_delta': round(float(self.put_delta), 3), 
    'put_vega': round(float(self.put_vega), 3), 'put_theta': round(float(self.put_theta), 3), 'put_rho': round(float(self.put_rho), 3)}


  @property
  def BS_params(self):
    return (self._s, self._sigma, self._k, self._t, self._r)
  
  @BS_params.setter
  def BS_params(self, *args):
    self._s, self._sigma, self._k, self._t, self._r = args[0]


  @staticmethod
  def __get_hist_data(STOCK_NAME: str) -> pd.DataFrame:
    CURR_DATE = datetime.today().date()
    START_DATE = CURR_DATE + timedelta(days=-365)
    hist_data = yf.download(STOCK_NAME, start=START_DATE, end=CURR_DATE)
    print(len(hist_data))
    
    return hist_data['Close']
  
  @staticmethod
  def __calc_annual_vol(hist_data: pd.DataFrame) -> float:
    daily_returns = jnp.array([hist_data[i + 1] / hist_data[i] for i in range(len(hist_data) - 1)])
    daily_returns = jnp.log(daily_returns)
    daily_vol = jnp.std(daily_returns)
    annual_vol = daily_vol * math.sqrt(252.0)
    
    return annual_vol
