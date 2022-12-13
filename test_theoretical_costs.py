import math as m 
import sympy as sy
PeL = 0
M = 750
a_penalty=0.001
T_f = 5*M
q_0=20

mu=-0.00000001
gamma=0.000001
V=100
eta=V*a_penalty                                                     
sigma=0.2

        
b=V*(pow(sigma,2))*gamma/(2*eta)
a=-(V*mu)/(2*eta)

q_now=q_0
cost_AC_drift=0
for i in range(5):
    next_t=(i+1)*M
    q_future =  ((q_0+(a/b))*(m.sinh(m.sqrt(b)*(T_f-next_t)))+(a/b)*m.sinh(m.sqrt(b)*next_t))/m.sinh(m.sqrt(b)*T_f)-(a/b)
    
    cost_AC_drift+=pow(100*(q_now-q_future),2)/M
    q_now=q_future
cost_AC_drift=a_penalty*cost_AC_drift

q_now=q_0
cost_AC_drift_limit=0
for k in range(5):
    next_t=(k+1)*M
    q_future=((T_f-next_t)*(4*eta*q_0+mu*V*next_t*T_f)/(4*eta*T_f))

    cost_AC_drift_limit+=pow(100*(q_now-q_future),2)/M
    q_now=q_future

cost_AC_drift_limit=a_penalty*cost_AC_drift_limit

cost_TWAP=a_penalty*pow(100*q_0,2)/(5*M)

print('Theoritical cost TWAP strategy = ', cost_TWAP)

print('Theoritical cost AC  with drift strategy = ', cost_AC_drift)

print('Theoritical cost AC with drift limit strategy = ', cost_AC_drift_limit)

q_now=q_0
p_0=150
PeL_AC=0
PeL_temporary=0
for k in range(5):
    next_t=(k+1)*M
    q_future=((T_f-next_t)*(4*eta*q_0+mu*V*next_t*T_f)/(4*eta*T_f))
    PeL_temporary=0
    for t in range(750):
        PeL_temporary+=(k*M+t)*mu+150
    print(q_now-q_future)
    PeL_AC+=100*(q_now-q_future)/M * PeL_temporary
    q_now=q_future
    
q_now=q_0
p_0=150
PeL_TWAP=0
PeL_temporary=0
for k in range(5):
    next_t=(k+1)*M
    q_future=q_now-q_0/5
    PeL_temporary=0
    for t in range(750):
        PeL_temporary+=(k*M+t)*mu+150
    PeL_TWAP+=100*(q_now-q_future)/M * PeL_temporary
    q_now=q_future
print(PeL_TWAP,PeL_AC)
print(PeL_TWAP-cost_TWAP,PeL_AC-cost_AC_drift_limit)

print(PeL_TWAP-cost_TWAP-PeL_AC+cost_AC_drift_limit)
theoretical_AC=100*q_0*150+ mu*((q_0*T_f)/2 + mu*pow(T_f,3)/(24*a_penalty))-a_penalty*(pow((100*q_0),2)/T_f + (pow(mu,2)*pow(T_f,3))/(48*pow(a_penalty,2)))
print(theoretical_AC)
theoretical_difference=3*pow(mu,2)*pow(T_f,3)*V/(48*eta)
print(theoretical_difference)
