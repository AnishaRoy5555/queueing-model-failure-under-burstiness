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
    
    return sim_results


# =============================================================================
# BURSTY ARRIVALS: When Poisson Assumptions Fail
# =============================================================================

class BurstyQueueSimulator(QueueSimulator):
    """
    Queue simulator with non-Poisson arrival processes.
    Demonstrates what happens when M/M/1 assumptions break.
    """
    
    def __init__(self, arrival_rate: float, service_rate: float, num_servers: int = 1,
                 arrival_type: str = "poisson", batch_mean: float = 1.0,
                 hawkes_alpha: float = 0.5, hawkes_beta: float = 1.0, seed: int = None):
        """
        Args:
            arrival_rate: Base arrival rate (λ)
            service_rate: Service rate (μ)
            num_servers: Number of servers (k)
            arrival_type: "poisson", "batch", or "hawkes"
            batch_mean: Mean batch size for batch arrivals
            hawkes_alpha: Jump size for Hawkes process (intensity increase per arrival)
            hawkes_beta: Decay rate for Hawkes process
            seed: Random seed
        """
        super().__init__(arrival_rate, service_rate, num_servers, seed)
        self.arrival_type = arrival_type
        self.batch_mean = batch_mean
        self.hawkes_alpha = hawkes_alpha
        self.hawkes_beta = hawkes_beta
        
        # For Hawkes process: track intensity history
        self.hawkes_arrivals = []  # List of arrival times
    
    def _generate_batch_size(self) -> int:
        """Generate batch size from geometric distribution with given mean."""
        # Geometric distribution: P(X = k) = (1-p)^(k-1) * p for k = 1, 2, 3, ...
        # Mean = 1/p, so p = 1/batch_mean
        # Using numpy-style: number of failures before first success + 1
        p = 1.0 / self.batch_mean
        # Generate geometrically distributed random variable
        k = 1
        while random.random() > p:
            k += 1
        return k
    
    def _hawkes_intensity(self, t: float) -> float:
        """
        Compute Hawkes intensity at time t.
        λ(t) = λ_base + Σ α * exp(-β * (t - t_i)) for all arrivals t_i < t
        """
        base = self.arrival_rate
        excitation = sum(
            self.hawkes_alpha * math.exp(-self.hawkes_beta * (t - ti))
            for ti in self.hawkes_arrivals if ti < t
        )
        return base + excitation
    
    def _generate_hawkes_interarrival(self) -> float:
        """
        Generate next arrival time using Ogata's thinning algorithm.
        """
        t = self.current_time
        
        while True:
            # Upper bound on intensity
            lambda_bar = self._hawkes_intensity(t) + self.hawkes_alpha
            
            # Generate candidate
            t += random.expovariate(lambda_bar)
            
            # Accept/reject
            lambda_t = self._hawkes_intensity(t)
            if random.random() < lambda_t / lambda_bar:
                return t - self.current_time
    
    def _handle_arrival(self, event: Event):
        """Process arrival - modified for batch arrivals."""
        
        if self.arrival_type == "batch":
            # Generate batch of customers
            batch_size = self._generate_batch_size()
            
            for i in range(batch_size):
                customer = Customer(
                    id=self.customer_counter + i,
                    arrival_time=event.time,
                    service_duration=self._generate_service_time()
                )
                self.customers.append(customer)
                
                # Try to start service
                idle_server = self._find_idle_server()
                if idle_server is not None:
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
                    self.waiting_queue.append(customer)
            
            self.customer_counter += batch_size
            
            # Schedule next batch arrival (Poisson between batches)
            next_arrival_time = self.current_time + self._generate_interarrival_time()
            self._schedule_event(Event(
                time=next_arrival_time,
                event_type=EventType.ARRIVAL,
                customer_id=self.customer_counter
            ))
            
        elif self.arrival_type == "hawkes":
            # Record arrival for Hawkes intensity
            self.hawkes_arrivals.append(event.time)
            
            # Standard single customer arrival
            customer = Customer(
                id=event.customer_id,
                arrival_time=event.time,
                service_duration=self._generate_service_time()
            )
            self.customers.append(customer)
            
            idle_server = self._find_idle_server()
            if idle_server is not None:
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
                self.waiting_queue.append(customer)
            
            self.customer_counter += 1
            
            # Schedule next arrival using Hawkes process
            next_arrival_time = self.current_time + self._generate_hawkes_interarrival()
            self._schedule_event(Event(
                time=next_arrival_time,
                event_type=EventType.ARRIVAL,
                customer_id=self.customer_counter
            ))
            
            # Prune old arrivals to prevent memory growth
            cutoff = self.current_time - 10.0 / self.hawkes_beta
            self.hawkes_arrivals = [t for t in self.hawkes_arrivals if t > cutoff]
            
        else:
            # Standard Poisson arrival
            super()._handle_arrival(event)


def run_bursty_comparison(arrival_rate: float, service_rate: float,
                          num_customers: int = 50000, seed: int = 42):
    """
    Compare Poisson vs Batch vs Hawkes arrivals.
    All have the same effective arrival rate for fair comparison.
    """
    
    print("=" * 70)
    print("ASSUMPTION FAILURE ANALYSIS: When Poisson Breaks")
    print("=" * 70)
    print(f"Base parameters: λ = {arrival_rate}, μ = {service_rate}")
    print(f"All scenarios calibrated to same effective arrival rate")
    print("-" * 70)
    
    # M/M/1 analytical baseline
    analytical = mm1_analytical(arrival_rate, service_rate)
    print(f"\nM/M/1 Analytical (Poisson assumption):")
    print(f"  E[W] = {analytical['E[W]']:.4f}")
    print(f"  E[Wq] = {analytical['E[Wq]']:.4f}")
    print(f"  E[L] = {analytical['E[L]']:.4f}")
    
    results = {}
    
    # 1. Baseline Poisson
    print(f"\n[1/3] Running Poisson (baseline)...")
    sim_poisson = BurstyQueueSimulator(
        arrival_rate, service_rate, 
        arrival_type="poisson", seed=seed
    )
    results["Poisson"] = sim_poisson.run(num_customers=num_customers, warmup_customers=5000)
    
    # 2. Batch arrivals (mean batch size = 2)
    # To keep same effective rate: reduce base rate by batch_mean
    batch_mean = 2.0
    adjusted_rate = arrival_rate / batch_mean
    print(f"[2/3] Running Batch arrivals (mean size = {batch_mean})...")
    sim_batch = BurstyQueueSimulator(
        adjusted_rate, service_rate,
        arrival_type="batch", batch_mean=batch_mean, seed=seed
    )
    results["Batch (μ=2)"] = sim_batch.run(num_customers=num_customers, warmup_customers=5000)
    
    # 3. Hawkes process
    # α/β ratio controls clustering intensity
    # Effective rate ≈ λ_base / (1 - α/β) when α < β
    hawkes_alpha = 0.4
    hawkes_beta = 1.0
    adjusted_base = arrival_rate * (1 - hawkes_alpha / hawkes_beta)
    print(f"[3/3] Running Hawkes process (α={hawkes_alpha}, β={hawkes_beta})...")
    sim_hawkes = BurstyQueueSimulator(
        adjusted_base, service_rate,
        arrival_type="hawkes", hawkes_alpha=hawkes_alpha, hawkes_beta=hawkes_beta,
        seed=seed
    )
    results["Hawkes"] = sim_hawkes.run(num_customers=num_customers, warmup_customers=5000)
    
    # Comparison table
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)
    print(f"{'Arrival Type':<15} {'E[Wq]':>10} {'E[W]':>10} {'E[L]':>10} {'vs M/M/1':>12}")
    print("-" * 70)
    
    baseline_w = analytical['E[W]']
    
    # Analytical row
    print(f"{'M/M/1 Theory':<15} {analytical['E[Wq]']:>10.4f} {analytical['E[W]']:>10.4f} {analytical['E[L]']:>10.4f} {'baseline':>12}")
    
    for name, res in results.items():
        pct_diff = (res['avg_time_in_system'] - baseline_w) / baseline_w * 100
        sign = "+" if pct_diff > 0 else ""
        print(f"{name:<15} {res['avg_wait_in_queue']:>10.4f} {res['avg_time_in_system']:>10.4f} {res['avg_system_length']:>10.4f} {sign}{pct_diff:>10.1f}%")
    
    print("=" * 70)
    
    # Key insights
    print("\nKEY INSIGHTS:")
    print("-" * 70)
    
    batch_increase = (results["Batch (μ=2)"]['avg_time_in_system'] - baseline_w) / baseline_w * 100
    hawkes_increase = (results["Hawkes"]['avg_time_in_system'] - baseline_w) / baseline_w * 100
    
    print(f"• Batch arrivals increase wait time by ~{batch_increase:.0f}% vs Poisson prediction")
    print(f"• Hawkes (self-exciting) increases wait time by ~{hawkes_increase:.0f}% vs Poisson prediction")
    print(f"• M/M/1 formulas UNDERESTIMATE congestion when arrivals are bursty")
    print(f"• This matters for: restaurant rushes, trading order flow, network traffic")
    print("=" * 70)
    
    return results


def run_bursty_comparison_mmk(arrival_rate: float, service_rate: float, num_servers: int,
                               num_customers: int = 50000, seed: int = 42):
    """
    Compare Poisson vs Batch vs Hawkes arrivals for M/M/k queue.
    """
    
    print("=" * 70)
    print(f"ASSUMPTION FAILURE ANALYSIS: M/M/{num_servers} with Bursty Arrivals")
    print("=" * 70)
    print(f"Parameters: λ = {arrival_rate}, μ = {service_rate}, k = {num_servers}")
    rho = arrival_rate / (num_servers * service_rate)
    print(f"Utilization: ρ = {rho:.4f}")
    print("-" * 70)
    
    # M/M/k analytical baseline
    analytical = mmk_analytical(arrival_rate, service_rate, num_servers)
    print(f"\nM/M/{num_servers} Analytical (Poisson assumption):")
    print(f"  E[W] = {analytical['E[W]']:.4f}")
    print(f"  E[Wq] = {analytical['E[Wq]']:.4f}")
    print(f"  P(wait) = {analytical['P(wait)']:.4f}")
    
    results = {}
    
    # 1. Baseline Poisson
    print(f"\n[1/3] Running Poisson (baseline)...")
    sim_poisson = BurstyQueueSimulator(
        arrival_rate, service_rate, num_servers,
        arrival_type="poisson", seed=seed
    )
    results["Poisson"] = sim_poisson.run(num_customers=num_customers, warmup_customers=5000)
    
    # 2. Batch arrivals
    batch_mean = 2.0
    adjusted_rate = arrival_rate / batch_mean
    print(f"[2/3] Running Batch arrivals (mean size = {batch_mean})...")
    sim_batch = BurstyQueueSimulator(
        adjusted_rate, service_rate, num_servers,
        arrival_type="batch", batch_mean=batch_mean, seed=seed
    )
    results["Batch (μ=2)"] = sim_batch.run(num_customers=num_customers, warmup_customers=5000)
    
    # 3. Hawkes process
    hawkes_alpha = 0.4
    hawkes_beta = 1.0
    adjusted_base = arrival_rate * (1 - hawkes_alpha / hawkes_beta)
    print(f"[3/3] Running Hawkes process (α={hawkes_alpha}, β={hawkes_beta})...")
    sim_hawkes = BurstyQueueSimulator(
        adjusted_base, service_rate, num_servers,
        arrival_type="hawkes", hawkes_alpha=hawkes_alpha, hawkes_beta=hawkes_beta,
        seed=seed
    )
    results["Hawkes"] = sim_hawkes.run(num_customers=num_customers, warmup_customers=5000)
    
    # Comparison table
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)
    print(f"{'Arrival Type':<15} {'E[Wq]':>10} {'E[W]':>10} {'P(wait)':>10} {'vs Theory':>12}")
    print("-" * 70)
    
    baseline_w = analytical['E[W]']
    
    # Analytical row
    print(f"{'M/M/k Theory':<15} {analytical['E[Wq]']:>10.4f} {analytical['E[W]']:>10.4f} {analytical['P(wait)']:>10.4f} {'baseline':>12}")
    
    for name, res in results.items():
        pct_diff = (res['avg_time_in_system'] - baseline_w) / baseline_w * 100
        sign = "+" if pct_diff > 0 else ""
        print(f"{name:<15} {res['avg_wait_in_queue']:>10.4f} {res['avg_time_in_system']:>10.4f} {res['prob_wait']:>10.4f} {sign}{pct_diff:>10.1f}%")
    
    print("=" * 70)
    
    return results, analytical


if __name__ == "__main__":
    # Test 1: M/M/1 with moderate load
    run_comparison(arrival_rate=0.8, service_rate=1.0, num_servers=1)
    
    # Test 2: M/M/1 with high load
    run_comparison(arrival_rate=0.95, service_rate=1.0, num_servers=1)
    
    # Test 3: M/M/3 with moderate load
    run_comparison(arrival_rate=2.0, service_rate=1.0, num_servers=3)
    
    # Test 4: M/M/5 with higher load
    run_comparison(arrival_rate=4.0, service_rate=1.0, num_servers=5)
    
    # Test 5: Bursty arrivals comparison
    print("\n" + "#" * 70)
    print("# PHASE 4: ASSUMPTION FAILURES")
    print("#" * 70 + "\n")
    
    # M/M/1 with bursty arrivals (ρ = 0.8)
    mm1_results = run_bursty_comparison(arrival_rate=0.8, service_rate=1.0)
    
    print("\n")
    
    # M/M/4 with bursty arrivals (same ρ = 0.8)
    # λ = 3.2 gives ρ = 3.2/(4×1.0) = 0.8
    mm4_results, mm4_analytical = run_bursty_comparison_mmk(
        arrival_rate=3.2, service_rate=1.0, num_servers=4
    )
    
    # Summary comparison
    print("\n" + "=" * 70)
    print("COMPARISON: Does M/M/k Handle Burstiness Better Than M/M/1?")
    print("=" * 70)
    print("Both systems at ρ = 0.8 utilization")
    print("-" * 70)
    
    mm1_theory = mm1_analytical(0.8, 1.0)
    mm1_batch_increase = (mm1_results["Batch (μ=2)"]['avg_time_in_system'] - mm1_theory['E[W]']) / mm1_theory['E[W]'] * 100
    mm1_hawkes_increase = (mm1_results["Hawkes"]['avg_time_in_system'] - mm1_theory['E[W]']) / mm1_theory['E[W]'] * 100
    
    mm4_batch_increase = (mm4_results["Batch (μ=2)"]['avg_time_in_system'] - mm4_analytical['E[W]']) / mm4_analytical['E[W]'] * 100
    mm4_hawkes_increase = (mm4_results["Hawkes"]['avg_time_in_system'] - mm4_analytical['E[W]']) / mm4_analytical['E[W]'] * 100
    
    print(f"{'System':<12} {'Batch Penalty':>15} {'Hawkes Penalty':>15}")
    print("-" * 70)
    print(f"{'M/M/1':<12} {'+' + f'{mm1_batch_increase:.1f}%':>15} {'+' + f'{mm1_hawkes_increase:.1f}%':>15}")
    print(f"{'M/M/4':<12} {'+' + f'{mm4_batch_increase:.1f}%':>15} {'+' + f'{mm4_hawkes_increase:.1f}%':>15}")
    print("=" * 70)
    print("\nConclusion:")
    print(f"  • M/M/1 batch penalty: +{mm1_batch_increase:.0f}%  |  M/M/4 batch penalty: +{mm4_batch_increase:.0f}%")
    print(f"  • M/M/1 Hawkes penalty: +{mm1_hawkes_increase:.0f}%  |  M/M/4 Hawkes penalty: +{mm4_hawkes_increase:.0f}%")
    print("  • Multiple servers REDUCE both absolute wait times AND relative burstiness penalty")
    print("  • This is because k servers can absorb short bursts before queue builds")
    print("=" * 70)
