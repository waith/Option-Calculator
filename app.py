import streamlit as st

from OptionPrice import OptionPrice

from functools import lru_cache

@lru_cache()
def get_option_calculators():
  call_op = OptionPrice(underlying_asset_current_price=1.0, strike_price=0.0, risk_free_rate=0.0, expiration_time=0.0, annual_vol=0.10, target_stock=None)
  put_op = OptionPrice(underlying_asset_current_price=1.0, strike_price=0.0, risk_free_rate=0.0, expiration_time=0.0, annual_vol=0.10, target_stock=None)
  return call_op, put_op

call_op, put_op = get_option_calculators()
st.title("Option Calculator 📊")

with st.sidebar:
  st.header("Information DashBoard")
  st.info("Risk-free rate and Implied Volatility should be entered in percent(%)")
  st.info("Time to expiration should be entered in number of days")


s = st.number_input("Underlying asset's price", -1.0, 
help="Current price of the target stock", value=-1.0, key='put_s')

k = st.number_input("Strike price", 0.0, help="interested strike price", 
value=0.0, key='put_k')

r = st.number_input("Risk-free interest rate (%)", 0.0, 
help="Risk-free interest rate (%)", value=10.0, key='put_r', disabled=True) / 100.0


put_col, call_col = st.columns(2)
with put_col:
  st.header("Put Calculator")
  


  sigma = st.number_input("Implied Volatility (%)", 0.0, 
  help="Implied Volatility of target stock", value=0.0, key='put_sigma') / 100.0

  t = int(st.number_input("Time to expiration (days)", 0, 
  help="Implied Volatility of target stock", value=0, key='put_t')) / 365.0

  ex_put = st.expander("View Option-greeks", expanded=True)
  with ex_put:
    if s != -1.0:
      put_op.BS_params = (s, sigma, k, t, r)
      put_op.put_greeks = put_op.calc_put_price
      st.write(put_op.put_greeks)

      st.metric("Price", value=f"{put_op.put_greeks['put_price']} (₹)")
      st.metric("Delta", value=f"{put_op.put_greeks['put_delta']}", delta=put_op.put_greeks['put_delta'])
      st.metric("Theta", value=f"{put_op.put_greeks['put_theta']}", delta=put_op.put_greeks['put_theta'])
      st.metric("Vega", value=f"{put_op.put_greeks['put_vega']}", delta=put_op.put_greeks['put_vega'])
      st.metric("Rho", value=f"{put_op.put_greeks['put_rho']}", delta=put_op.put_greeks['put_rho'])


with call_col:
  st.header("Call Calculator")

  sigma = st.number_input("Implied Volatility (%)", 0.0, 
  help="Implied Volatility of target stock", value=0.0, key='call_sigma') / 100.0

  t = int(st.number_input("Time to expiration (days)", 0, 
  help="Implied Volatility of target stock", value=0, key='call_t')) / 365.0

  ex_call = st.expander("View Option-greeks", expanded=True)
  with ex_call:
    if s != -1.0:
      call_op.BS_params = (s, sigma, k, t, r)
      call_op.call_greeks = call_op.calc_call_price
      st.write(call_op.call_greeks)

      st.metric("Price", value=f"{call_op.call_greeks['call_price']} (₹)")
      st.metric("Delta", value=f"{call_op.call_greeks['call_delta']}", delta=call_op.call_greeks['call_delta'])
      st.metric("Theta", value=f"{call_op.call_greeks['call_theta']}", delta=call_op.call_greeks['call_theta'])
      st.metric("Vega", value=f"{call_op.call_greeks['call_vega']}", delta=call_op.call_greeks['call_vega'])
      st.metric("Rho", value=f"{call_op.call_greeks['call_rho']}", delta=call_op.call_greeks['call_rho'])
