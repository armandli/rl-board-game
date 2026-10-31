#ifndef RLBG_EVALUATION_ELO_H
#define RLBG_EVALUATION_ELO_H

#include <cmath>

#include <map>
#include <set>
#include <string>
#include <utility>

#include <evaluation/arena.h>

namespace rlbg {

namespace s = std;

// Elo ratings of every player from all match results at once, fit as a
// Bradley-Terry model (a draw counts as half a win for each side) with the
// minorization-maximization algorithm. Each pair that met also gets one
// virtual draw, so a player that never loses keeps a finite rating.
struct EloTable {
  void add(const s::string& a, const s::string& b, const MatchResult& r){
    mPlayers.insert(a);
    mPlayers.insert(b);
    auto& ab = mScores[{a, b}];
    ab.first += r.wins + 0.5 * r.draws;
    ab.second += r.games();
    auto& ba = mScores[{b, a}];
    ba.first += r.losses + 0.5 * r.draws;
    ba.second += r.games();
  }

  // ratings with anchor fixed at anchor_rating; players not connected to
  // the anchor through matches float freely
  s::map<s::string, double> ratings(const s::string& anchor, double anchor_rating = 0.) const {
    s::map<s::string, double> gamma;
    for (const s::string& p : mPlayers) gamma[p] = 1.;
    for (int it = 0; it < 500; ++it){
      double change = 0.;
      for (const s::string& p : mPlayers){
        double wins = 0.;
        double denom = 0.;
        for (const auto& [key, wn] : mScores){
          if (key.first != p) continue;
          double n = wn.second + 1.;
          wins += wn.first + 0.5;
          denom += n / (gamma[p] + gamma[key.second]);
        }
        if (denom <= 0.) continue;
        double g = wins / denom;
        change = s::max(change, s::abs(s::log(g / gamma[p])));
        gamma[p] = g;
      }
      if (change < 1e-7) break;
    }
    double ref = gamma.contains(anchor) ? gamma[anchor] : 1.;
    s::map<s::string, double> out;
    for (const auto& [p, g] : gamma) out[p] = anchor_rating + 400. * s::log10(g / ref);
    return out;
  }

protected:
  s::set<s::string>                                       mPlayers;
  // (a, b) -> (score of a against b, games)
  s::map<s::pair<s::string, s::string>, s::pair<double, double>> mScores;
};

}  // namespace rlbg

#endif//RLBG_EVALUATION_ELO_H
