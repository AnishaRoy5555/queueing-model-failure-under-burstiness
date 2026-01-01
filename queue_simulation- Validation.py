"""
Discrete-Event Simulation for M/M/1 and M/M/k Queues
=====================================================

Validates analytical results:
- M/M/1: E[W] = 1/(μ-λ), E[Wq] = ρ/(μ-λ), E[L] = ρ/(1-ρ)
- M/M/k: Erlang-C formula for P(wait), E[Wq] = P(wait)/(kμ-λ)
"""

import heapq
import random
import math
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class EventType(Enum):
    ARRIVAL = 1
    DEPARTURE = 2


@dataclass(order=True)
class Event:
    time: float
    event_type: EventType = field(compare=False)
    customer_id: int = field(compare=False)
    server_id: Optional[int] = field(default=None, compare=False)


@dataclass
class Customer:
    id: int
    arrival_time: float
    service_duration: float
    service_start_time: Optional[float] = None
    departure_time: Optional[float] = None
    
    @property
    def wait_in_queue(self) -> float:
        if self.service_start_time is None:
            return 0.0
        return self.service_start_time - self.arrival_time
    
    @property
    def total_time_in_system(self) -> float:
        if self.departure_time is None:
            return 0.0
        return self.departure_time - self.arrival_time


class QueueSimulator:
    def __init__(self, arrival_rate: float, service_rate: float, num_servers: int = 1, seed: int = None):
        """
        Initialize M/M/k queue simulator.
        
        Args:
            arrival_rate: λ - Poisson arrival rate
            service_rate: μ - Exponential service rate (per server)
            num_servers: k - Number of parallel servers
            seed: Random seed for reproducibility
        """
        self.arrival_rate = arrival_rate  # λ
        self.service_rate = service_rate  # μ
        self.num_servers = num_servers    # k
        
        if seed is not None:
            random.seed(seed)
        
        # State
        self.current_time = 0.0
        self.event_queue: List[Event] = []
        self.waiting_queue: List[Customer] = []
        self.server_busy_until: List[float] = [0.0] * num_servers  # Time each server becomes free
        
        # Tracking
        self.customers: List[Customer] = []
        self.customer_counter = 0
        
        # For time-averaged queue length
        self.last_state_change_time = 0.0
        self.queue_length_integral = 0.0
        self.system_length_integral = 0.0
    
    def _generate_interarrival_time(self) -> float:
        """Sample from Exp(λ)."""
        return random.expovariate(self.arrival_rate)
    
    def _generate_service_time(self) -> float:
        """Sample from Exp(μ)."""
        return random.expovariate(self.service_rate)
    
    def _find_idle_server(self) -> Optional[int]:
        """Find a server that is idle at current_time. Returns None if all busy."""
        for i, busy_until in enumerate(self.server_busy_until):
            if busy_until <= self.current_time:
                return i
        return None
    
    def _customers_in_system(self) -> int:
        """Current number of customers in system (in queue + in service)."""
        in_service = sum(1 for t in self.server_busy_until if t > self.current_time)
        return len(self.waiting_queue) + in_service
    
    def _update_integrals(self, new_time: float):
        """Update time-weighted integrals for average calculations."""
        dt = new_time - self.last_state_change_time
        if dt > 0:
            self.queue_length_integral += len(self.waiting_queue) * dt
            self.system_length_integral += self._customers_in_system() * dt
        self.last_state_change_time = new_time
    
    def _schedule_event(self, event: Event):
        """Add event to priority queue."""
        heapq.heappush(self.event_queue, event)
    
    def _handle_arrival(self, event: Event):
        """Process an arrival event."""
        # Create customer
        customer = Customer(
            id=event.customer_id,
            arrival_time=event.time,
            service_duration=self._generate_service_time()
        )
        self.customers.append(customer)
        
        # Schedule next arrival
        self.customer_counter += 1
        next_arrival_time = self.current_time + self._generate_interarrival_time()
        self._schedule_event(Event(
            time=next_arrival_time,
            event_type=EventType.ARRIVAL,
            customer_id=self.customer_counter
        ))
        
        # Try to start service immediately
        idle_server = self._find_idle_server()
        if idle_server is not None:
            # Start service immediately
            customer.service_start_time = self.current_time
            finish_time = self.current_time + customer.service_duration
            self.server_busy_until[idle_server] = finish_time
            self._schedule_event(Event(
                time=finish_time,
                event_type=EventType.DEPARTURE,
                customer_id=customer.id,
                server_id=idle_server
            ))
        else:
            # All servers busy, join queue
            self.waiting_queue.append(customer)
    
    def _handle_departure(self, event: Event):
        """Process a departure event."""
        # Find the departing customer and record departure
        for customer in self.customers:
            if customer.id == event.customer_id:
                customer.departure_time = event.time
                break
        
        # If queue is non-empty, start serving next customer
        if self.waiting_queue:
            next_customer = self.waiting_queue.pop(0)
            next_customer.service_start_time = self.current_time
            finish_time = self.current_time + next_customer.service_duration
            self.server_busy_until[event.server_id] = finish_time
            self._schedule_event(Event(
                time=finish_time,
                event_type=EventType.DEPARTURE,
                customer_id=next_customer.id,
                server_id=event.server_id
            ))
    
    def run(self, num_customers: int = 10000, warmup_customers: int = 1000) -> dict:
        """
        Run simulation until num_customers have been served.
        
        Args:
            num_customers: Total customers to process (including warmup)
            warmup_customers: Customers to discard for steady-state
        
        Returns:
            Dictionary of performance metrics
        """
        # Schedule first arrival
        self.customer_counter = 0
        first_arrival_time = self._generate_interarrival_time()
        self._schedule_event(Event(
            time=first_arrival_time,
            event_type=EventType.ARRIVAL,
            customer_id=self.customer_counter
        ))
        self.customer_counter += 1
        
        # Main simulation loop
        customers_departed = 0
        while customers_departed < num_customers and self.event_queue:
            event = heapq.heappop(self.event_queue)
            
            # Update time-weighted integrals before state change
            self._update_integrals(event.time)
            self.current_time = event.time
            
            if event.event_type == EventType.ARRIVAL:
                self._handle_arrival(event)
            else:
                self._handle_departure(event)
                customers_departed += 1
        
        # Compute metrics (excluding warmup)
        completed_customers = [c for c in self.customers 
                              if c.departure_time is not None 
                              and c.id >= warmup_customers]
        
        if not completed_customers:
            return {"error": "No customers completed after warmup"}
        
        wait_times = [c.wait_in_queue for c in completed_customers]
        system_times = [c.total_time_in_system for c in completed_customers]
        
        return {
            "num_customers": len(completed_customers),
            "avg_wait_in_queue": sum(wait_times) / len(wait_times),
            "avg_time_in_system": sum(system_times) / len(system_times),
            "avg_queue_length": self.queue_length_integral / self.current_time,
            "avg_system_length": self.system_length_integral / self.current_time,
            "simulation_time": self.current_time,
            "prob_wait": sum(1 for w in wait_times if w > 0) / len(wait_times)
        }


def mm1_analytical(arrival_rate: float, service_rate: float) -> dict:
    """Compute analytical M/M/1 results."""
    lam, mu = arrival_rate, service_rate
    rho = lam / mu
    
    if rho >= 1:
        return {"error": "System unstable (ρ >= 1)"}
    
    return {
        "rho": rho,
        "E[L]": rho / (1 - rho),
        "E[W]": 1 / (mu - lam),
        "E[Wq]": rho / (mu - lam),
        "E[Lq]": (rho ** 2) / (1 - rho)
    }


def mmk_analytical(arrival_rate: float, service_rate: float, num_servers: int) -> dict:
    """Compute analytical M/M/k results using Erlang-C formula."""
    lam, mu, k = arrival_rate, service_rate, num_servers
    rho = lam / (k * mu)  # Utilization per server
    a = lam / mu          # Offered load (k * rho)
    
    if rho >= 1:
        return {"error": "System unstable (ρ >= 1)"}
    
    # Compute π_0
    # Sum for n = 0 to k-1: (kρ)^n / n! = a^n / n!
    finite_sum = sum((a ** n) / math.factorial(n) for n in range(k))
    
    # Tail term: (kρ)^k / (k! * (1-ρ)) = a^k / (k! * (1-ρ))
    tail_term = (a ** k) / (math.factorial(k) * (1 - rho))
    
    pi_0 = 1 / (finite_sum + tail_term)
    
    # Erlang-C: P(wait)
    p_wait = tail_term * pi_0
    
    # Expected queue length
    E_Lq = p_wait * rho / (1 - rho)
    
    # Expected wait in queue (Little's Law)
    E_Wq = p_wait / (k * mu - lam)
    
    # Expected time in system
    E_W = E_Wq + 1 / mu
    
    # Expected number in system
    E_L = E_Lq + a
    
    return {
        "rho": rho,
        "pi_0": pi_0,
        "P(wait)": p_wait,
        "E[Lq]": E_Lq,
        "E[L]": E_L,
        "E[Wq]": E_Wq,
        "E[W]": E_W
    }


def run_comparison(arrival_rate: float, service_rate: float, num_servers: int = 1,
                   num_customers: int = 50000, seed: int = 42):
    """Run simulation and compare to analytical results."""
    
    print("=" * 70)
    if num_servers == 1:
        print(f"M/M/1 Queue Comparison")
    else:
        print(f"M/M/{num_servers} Queue Comparison")
    print("=" * 70)
    print(f"Parameters: λ = {arrival_rate}, μ = {service_rate}, k = {num_servers}")
    print(f"Utilization: ρ = {arrival_rate / (num_servers * service_rate):.4f}")
    print("-" * 70)
    
    # Analytical results
    if num_servers == 1:
        analytical = mm1_analytical(arrival_rate, service_rate)
    else:
        analytical = mmk_analytical(arrival_rate, service_rate, num_servers)
    
    if "error" in analytical:
        print(f"Analytical: {analytical['error']}")
        return
    
    print("\nAnalytical Results:")
    for key, value in analytical.items():
        print(f"  {key}: {value:.6f}")
    
    # Simulation results
    print(f"\nRunning simulation with {num_customers} customers...")
    sim = QueueSimulator(arrival_rate, service_rate, num_servers, seed=seed)
    sim_results = sim.run(num_customers=num_customers, warmup_customers=5000)
    
    print("\nSimulation Results:")
    for key, value in sim_results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    
    # Comparison
    print("\nComparison (Simulation vs Analytical):")
    print("-" * 70)
    
    comparisons = [
        ("E[Wq]", sim_results["avg_wait_in_queue"], analytical.get("E[Wq]")),
        ("E[W]", sim_results["avg_time_in_system"], analytical.get("E[W]")),
        ("E[L]", sim_results["avg_system_length"], analytical.get("E[L]")),
    ]
    
    if num_servers > 1:
        comparisons.append(("P(wait)", sim_results["prob_wait"], analytical.get("P(wait)")))
    
    print(f"{'Metric':<12} {'Simulation':>14} {'Analytical':>14} {'Error %':>12}")
    print("-" * 70)
    for name, sim_val, ana_val in comparisons:
        if ana_val is not None and ana_val != 0:
            error_pct = abs(sim_val - ana_val) / ana_val * 100
            print(f"{name:<12} {sim_val:>14.6f} {ana_val:>14.6f} {error_pct:>11.2f}%")
        else:
            print(f"{name:<12} {sim_val:>14.6f} {'N/A':>14}")
    
    print("=" * 70)
    print()


if __name__ == "__main__":
    # Test 1: M/M/1 with moderate load
    run_comparison(arrival_rate=0.8, service_rate=1.0, num_servers=1)
    
    # Test 2: M/M/1 with high load
    run_comparison(arrival_rate=0.95, service_rate=1.0, num_servers=1)
    
    # Test 3: M/M/3 with moderate load
    run_comparison(arrival_rate=2.0, service_rate=1.0, num_servers=3)
    
    # Test 4: M/M/5 with higher load
    run_comparison(arrival_rate=4.0, service_rate=1.0, num_servers=5)
