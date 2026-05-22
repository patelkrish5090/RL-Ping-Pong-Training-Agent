"""
Episode Analyzer for Anti-Jamming Channel Selection
=====================================================
Replaces the Pong hit/miss/distance analyzer with wireless communication
metrics relevant to the MILCOM anti-jamming research context.

Tracked metrics per episode:
  - Packet Delivery Ratio (PDR)
  - Average throughput (packets/step)
  - Jammed transmission rate
  - Average queue length
  - Channel switching rate (switches/step)
  - Energy cost
  - Adaptation behavior under jammer changes
"""

import numpy as np
from typing import List, Dict, Optional


class EpisodeAnalyzer:
    """
    Analyzes training episodes to generate meaningful feedback for the LLM.
    Tracks statistics across episodes and generates behavior descriptions
    in the wireless anti-jamming domain.
    """

    def __init__(self):
        self.episodes: List[Dict] = []
        self.step_data: List[Dict] = []  # Detailed step-by-step data (optional)

    def add_episode(self, stats: Dict):
        """
        Add episode statistics.

        Args:
            stats: Dictionary containing episode metrics, expected keys:
                - pdr:         Packet Delivery Ratio (0–1)
                - throughput:  Packets delivered per step
                - jammed_rate: Fraction of TX attempts that were jammed
                - avg_queue:   Average queue length during episode
                - switches:    Total channel switches
                - switch_rate: Switches per step
                - energy:      Total energy consumed
                - total_steps: Episode length
                - total_delivered: Total packets delivered
                - total_tx:    Total transmission attempts
        """
        self.episodes.append(stats)

    def add_step(self, step_info: Dict):
        """Add step-level data for detailed analysis (optional)."""
        self.step_data.append(step_info)

    def clear(self):
        """Clear stored episodes and step data."""
        self.episodes = []
        self.step_data = []

    def analyze(self, n_episodes: Optional[int] = None) -> Dict:
        """
        Analyze last N episodes and compute aggregate wireless metrics.

        Args:
            n_episodes: Number of recent episodes to analyze (None = all)

        Returns:
            Dictionary with statistics and behavior analysis
        """
        if n_episodes is not None:
            recent = self.episodes[-n_episodes:]
        else:
            recent = self.episodes

        if not recent:
            return {
                "error": "No episodes to analyze",
                "n_episodes": 0,
                "avg_pdr": 0.0,
                "avg_score": 0.0,  # Alias for train_iterative compatibility
                "behaviors": "No data available — agent has not completed any episodes yet.",
            }

        n = len(recent)

        # ---- Core wireless metrics ----------------------------------------
        pdrs         = [ep.get("pdr", 0.0)         for ep in recent]
        throughputs  = [ep.get("throughput", 0.0)   for ep in recent]
        jammed_rates = [ep.get("jammed_rate", 0.0)  for ep in recent]
        avg_queues   = [ep.get("avg_queue", 0.0)    for ep in recent]
        switch_rates = [ep.get("switch_rate", 0.0)  for ep in recent]
        energies     = [ep.get("energy", 0.0)       for ep in recent]
        lengths      = [ep.get("total_steps", ep.get("length", 0)) for ep in recent]

        analysis = {
            "n_episodes":       n,
            "avg_pdr":          float(np.mean(pdrs)),
            "best_pdr":         float(np.max(pdrs)),
            "worst_pdr":        float(np.min(pdrs)),
            "pdr_std":          float(np.std(pdrs)),
            "avg_throughput":   float(np.mean(throughputs)),
            "avg_jammed_rate":  float(np.mean(jammed_rates)),
            "avg_queue":        float(np.mean(avg_queues)),
            "avg_switch_rate":  float(np.mean(switch_rates)),
            "avg_energy":       float(np.mean(energies)),
            "avg_length":       float(np.mean(lengths)) if lengths else 0.0,
            # Alias used by train_iterative.py for convergence check
            "avg_score":        float(np.mean(pdrs)),
            "best_score":       float(np.max(pdrs)),
            "worst_score":      float(np.min(pdrs)),
        }

        # High-PDR episodes (PDR >= 0.75 = "good performance")
        good_eps = sum(1 for p in pdrs if p >= 0.75)
        analysis["good_episode_rate"] = (good_eps / n) * 100

        # Generate behavior description
        analysis["behaviors"] = self._describe_behaviors(analysis)

        return analysis

    def _describe_behaviors(self, stats: Dict) -> str:
        """Generate natural language description of agent behavior for LLM."""
        behaviors = []

        avg_pdr        = stats.get("avg_pdr", 0.0)
        avg_jammed     = stats.get("avg_jammed_rate", 0.0)
        avg_queue      = stats.get("avg_queue", 0.0)
        avg_switch     = stats.get("avg_switch_rate", 0.0)
        good_rate      = stats.get("good_episode_rate", 0.0)
        avg_throughput = stats.get("avg_throughput", 0.0)

        # ---- PDR assessment -----------------------------------------------
        if avg_pdr < 0.30:
            behaviors.append("[FAIL] Very low PDR (<30%) -- agent is selecting jammed channels frequently")
        elif avg_pdr < 0.50:
            behaviors.append("[WARN] Low PDR (30-50%) -- agent is partially avoiding jammer but inconsistently")
        elif avg_pdr < 0.70:
            behaviors.append("[OK]   Moderate PDR (50-70%) -- agent avoids jamming in most episodes")
        elif avg_pdr < 0.85:
            behaviors.append("[GOOD] Good PDR (70-85%) -- effective channel selection with room to improve")
        else:
            behaviors.append("[BEST] Excellent PDR (>85%) -- agent reliably avoids jamming and delivers packets")

        # ---- Jammed TX rate -----------------------------------------------
        if avg_jammed > 0.50:
            behaviors.append("[HIGH]  High jammed TX rate (>50%) -- agent is not avoiding jammer effectively")
        elif avg_jammed > 0.30:
            behaviors.append("[MED]   Moderate jammed TX rate (30-50%) -- still transmitting on jammed channels")
        elif avg_jammed > 0.10:
            behaviors.append("[LOW]   Low jammed TX rate (10-30%) -- occasional jamming collisions")
        else:
            behaviors.append("[CLEAR] Minimal jammed TX rate (<10%) -- excellent jammer avoidance")

        # ---- Queue buildup ------------------------------------------------
        if avg_queue > 15:
            behaviors.append("[QUEUE-HIGH] Severe queue buildup (>15 pkts) -- throughput is far below arrival rate")
        elif avg_queue > 10:
            behaviors.append("[QUEUE-MED]  Moderate queue buildup (10-15 pkts) -- delivery not keeping up with arrivals")
        elif avg_queue > 5:
            behaviors.append("[QUEUE-LOW]  Mild queue pressure (5-10 pkts) -- manageable but worth reducing")
        else:
            behaviors.append("[QUEUE-OK]   Low queue depth (<5 pkts) -- agent is clearing the queue effectively")

        # ---- Channel switching behavior -----------------------------------
        if avg_switch > 0.50:
            behaviors.append("[SWITCH-HIGH] Very high switching rate (>50%/step) -- excessive channel hopping, "
                             "wasting energy and causing instability")
        elif avg_switch > 0.25:
            behaviors.append("[SWITCH-MED]  High switching rate (25-50%/step) -- more switches than necessary")
        elif avg_switch > 0.10:
            behaviors.append("[SWITCH-OK]   Moderate switching rate (10-25%/step) -- reasonable adaptation behavior")
        else:
            behaviors.append("[SWITCH-LOW]  Low switching rate (<10%/step) -- agent tends to stay on one channel; "
                             "may be too slow to adapt to jammer changes")

        # ---- Throughput ---------------------------------------------------
        if avg_throughput < 0.10:
            behaviors.append("[THRU-CRIT] Very low throughput (<0.10 pkts/step) -- nearly no packets getting through")
        elif avg_throughput < 0.30:
            behaviors.append("[THRU-LOW]  Low throughput (0.10-0.30 pkts/step)")
        elif avg_throughput < 0.50:
            behaviors.append("[THRU-MED]  Moderate throughput (0.30-0.50 pkts/step)")
        else:
            behaviors.append("[THRU-HIGH] High throughput (>0.50 pkts/step) -- strong channel utilization")

        # ---- Actionable recommendations -----------------------------------
        if avg_pdr < 0.40 and avg_jammed > 0.40:
            behaviors.append("\nRECOMMENDATION: Agent needs stronger incentive to avoid jammed channels. "
                             "Add penalty for selecting recently-jammed channels.")
        elif avg_queue > 12 and avg_switch < 0.15:
            behaviors.append("\nRECOMMENDATION: Queue is growing -- agent may be stuck on a low-quality "
                             "channel. Reward switching when queue pressure is high.")
        elif avg_switch > 0.40 and avg_pdr > 0.60:
            behaviors.append("\nRECOMMENDATION: Excessive switching despite decent PDR. "
                             "Add a switching penalty to reduce unnecessary hopping.")
        elif avg_pdr > 0.70 and avg_jammed < 0.15:
            behaviors.append("\nRECOMMENDATION: Good avoidance behavior. Fine-tune reward "
                             "to optimize energy efficiency and throughput consistency.")

        return "\n".join(f"- {b}" if not b.startswith("\n") else b for b in behaviors)

    def get_summary_for_llm(self, n_episodes: int = 20) -> str:
        """
        Get a formatted summary string suitable for the LLM prompt.

        Args:
            n_episodes: Number of recent episodes to summarize

        Returns:
            Formatted string for LLM
        """
        stats = self.analyze(n_episodes)

        summary = f"""## Training Results — Anti-Jamming Channel Selection (Last {stats['n_episodes']} Episodes)

### Packet Delivery & Throughput
- Average PDR (Packet Delivery Ratio): {stats['avg_pdr']:.3f}
- Best Episode PDR: {stats['best_pdr']:.3f}
- Worst Episode PDR: {stats['worst_pdr']:.3f}
- PDR Std Dev: {stats['pdr_std']:.3f}
- Average Throughput: {stats['avg_throughput']:.3f} pkts/step
- Good Episode Rate (PDR >= 0.75): {stats['good_episode_rate']:.1f}%

### Anti-Jamming Behavior
- Average Jammed TX Rate: {stats['avg_jammed_rate']:.3f}
- Average Queue Length: {stats['avg_queue']:.2f} packets

### Channel Management
- Average Switching Rate: {stats['avg_switch_rate']:.3f} switches/step
- Average Energy per Episode: {stats['avg_energy']:.1f} units
- Average Episode Length: {stats['avg_length']:.0f} steps

### Observed Behaviors
{stats['behaviors']}
"""
        return summary
