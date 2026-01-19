"""Analyze episodes and generate feedback for LLM"""
import numpy as np
from typing import List, Dict, Optional
from collections import defaultdict


class EpisodeAnalyzer:
    """
    Analyzes training episodes to generate meaningful feedback for LLM.
    Tracks statistics across episodes and generates behavior descriptions.
    """
    
    def __init__(self):
        self.episodes: List[Dict] = []
        self.step_data: List[Dict] = []  # Detailed step-by-step data
    
    def add_episode(self, stats: Dict):
        """
        Add episode statistics.
        
        Args:
            stats: Dictionary containing episode data like:
                - total_reward: Episode reward
                - length: Episode length
                - hits: Number of successful ball returns
                - misses: Number of missed balls
                - paddle_distances: List of paddle-ball distances
        """
        self.episodes.append(stats)
    
    def add_step(self, step_info: Dict):
        """Add step-level data for detailed analysis"""
        self.step_data.append(step_info)
    
    def clear(self):
        """Clear stored episodes and step data"""
        self.episodes = []
        self.step_data = []
    
    def analyze(self, n_episodes: Optional[int] = None) -> Dict:
        """
        Analyze last N episodes and generate LLM feedback.
        
        Args:
            n_episodes: Number of episodes to analyze (None = all)
        
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
                "avg_score": -21,
                "behaviors": "No data available - agent hasn't played any episodes yet.",
            }
        
        # Basic statistics
        scores = [ep.get("total_reward", ep.get("reward", 0)) for ep in recent]
        lengths = [ep.get("length", ep.get("l", 0)) for ep in recent]
        
        analysis = {
            "n_episodes": len(recent),
            "avg_score": float(np.mean(scores)),
            "best_score": float(max(scores)),
            "worst_score": float(min(scores)),
            "score_std": float(np.std(scores)),
            "avg_length": float(np.mean(lengths)) if lengths else 0,
        }
        
        # Win rate (positive score = winning)
        wins = sum(1 for s in scores if s > 0)
        analysis["win_rate"] = (wins / len(scores)) * 100 if scores else 0
        
        # Paddle-ball distance analysis
        all_distances = []
        for ep in recent:
            distances = ep.get("paddle_distances", ep.get("paddle_ball_distances", []))
            if distances:
                all_distances.extend(distances)
        
        if all_distances:
            analysis["avg_paddle_distance"] = float(np.mean(all_distances))
            analysis["min_paddle_distance"] = float(np.min(all_distances))
            analysis["max_paddle_distance"] = float(np.max(all_distances))
        else:
            analysis["avg_paddle_distance"] = 50.0  # Default estimate
        
        # Hit/miss analysis
        total_hits = sum(ep.get("hits", 0) for ep in recent)
        total_misses = sum(ep.get("misses", 0) for ep in recent)
        analysis["total_hits"] = total_hits
        analysis["total_misses"] = total_misses
        
        if total_hits + total_misses > 0:
            analysis["hit_rate"] = (total_hits / (total_hits + total_misses)) * 100
        else:
            analysis["hit_rate"] = 0.0
        
        # Generate behavior description
        analysis["behaviors"] = self._describe_behaviors(analysis)
        
        return analysis
    
    def _describe_behaviors(self, stats: Dict) -> str:
        """Generate natural language description of agent behavior for LLM"""
        behaviors = []
        
        avg_score = stats.get("avg_score", -21)
        avg_distance = stats.get("avg_paddle_distance", 100)
        hit_rate = stats.get("hit_rate", 0)
        win_rate = stats.get("win_rate", 0)
        
        # Score-based observations
        if avg_score < -15:
            behaviors.append("❌ Agent is losing badly (score < -15) - not tracking ball effectively")
        elif avg_score < -5:
            behaviors.append("⚠️ Agent loses more than wins - tracking is inconsistent")
        elif avg_score < 5:
            behaviors.append("➡️ Agent is competitive - basic tracking works but needs improvement")
        elif avg_score < 15:
            behaviors.append("✓ Agent is winning - good ball tracking established")
        else:
            behaviors.append("✅ Agent is dominating - excellent performance")
        
        # Distance-based observations
        if avg_distance > 60:
            behaviors.append("🔴 Paddle very far from ball (>60px) - needs fundamental tracking improvement")
        elif avg_distance > 40:
            behaviors.append("🟠 Paddle often far from ball (40-60px) - tracking too slow")
        elif avg_distance > 20:
            behaviors.append("🟡 Paddle moderately close (20-40px) - can improve anticipation")
        else:
            behaviors.append("🟢 Paddle stays close to ball (<20px) - good alignment")
        
        # Hit rate observations
        if hit_rate < 20:
            behaviors.append("Miss Rate: Very high (>80%) - agent misses most incoming balls")
        elif hit_rate < 40:
            behaviors.append("Miss Rate: High (60-80%) - agent misses many balls")
        elif hit_rate < 60:
            behaviors.append("Miss Rate: Moderate (40-60%) - room for improvement")
        else:
            behaviors.append("Hit Rate: Good (>60%) - successfully returns most balls")
        
        # Specific recommendations based on patterns
        if avg_score < -10 and avg_distance > 40:
            behaviors.append("\n📋 RECOMMENDATION: Focus reward on paddle-ball Y alignment")
        elif avg_score < 0 and avg_distance < 30:
            behaviors.append("\n📋 RECOMMENDATION: Agent tracks but reacts too slowly - reward quick movements")
        elif avg_score > 0 and win_rate < 70:
            behaviors.append("\n📋 RECOMMENDATION: Add strategic positioning reward for offensive play")
        
        return "\n".join(f"- {b}" if not b.startswith("\n") else b for b in behaviors)
    
    def get_summary_for_llm(self, n_episodes: int = 10) -> str:
        """
        Get a formatted summary string suitable for LLM prompt.
        
        Args:
            n_episodes: Number of recent episodes to summarize
        
        Returns:
            Formatted string for LLM
        """
        stats = self.analyze(n_episodes)
        
        summary = f"""## Training Results (Last {stats['n_episodes']} Episodes)

### Performance Metrics
- Average Score: {stats['avg_score']:.2f}
- Best Score: {stats['best_score']:.0f}
- Worst Score: {stats['worst_score']:.0f}
- Win Rate: {stats['win_rate']:.1f}%
- Average Episode Length: {stats['avg_length']:.0f} steps

### Ball Tracking
- Average Paddle-Ball Distance: {stats.get('avg_paddle_distance', 'N/A'):.1f} pixels
- Hit Rate: {stats.get('hit_rate', 0):.1f}%
- Total Hits: {stats['total_hits']}
- Total Misses: {stats['total_misses']}

### Observed Behaviors
{stats['behaviors']}
"""
        return summary
