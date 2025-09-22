import random
import math

# Assurez-vous d'avoir importé MPC_FIELD_SIZE et MPC_THRESHOLD depuis config
from config import MPC_FIELD_SIZE, MPC_THRESHOLD

class ShamirSecretSharing:
    """Implémentation robuste du partage de secret de Shamir"""

    def __init__(self, threshold=MPC_THRESHOLD, prime=None):
        self.threshold = int(threshold)
        self.prime = int(prime) if prime is not None else int(MPC_FIELD_SIZE)
        self._scale = 1_000_000  # facteur d'échelle pour floats -> int

        # Vérifier que prime est premier; si non, essayer de trouver le suivant (simple)
        if not self._is_probable_prime(self.prime):
            orig = self.prime
            self.prime = self._next_prime(self.prime)
            print(f"[Shamir] MPC_FIELD_SIZE {orig} n'était pas premier. Utilisation de {self.prime} à la place.")

        # sécurité: prime doit être > threshold et > scale
        if self.prime <= self.threshold or self.prime <= self._scale:
            raise ValueError("Le prime choisi est trop petit; prenez un MPC_FIELD_SIZE beaucoup plus grand.")

    # --------- API publique ---------
    def share_secret(self, secret, num_shares):
        """Retourne une liste (xi, yi) de shares. xi distincts dans [1, prime-1]."""
        secret_int = int(round(float(secret) * self._scale)) % self.prime
        if secret_int == 0:
            # éviter 0 car peut compliquer certains calculs ; on encode 0 --> prime-1
            secret_int = self.prime - 1

        # coefficients aléatoires du polynôme (deg = threshold-1)
        coefficients = [secret_int] + [random.randrange(1, self.prime - 1) for _ in range(self.threshold - 1)]

        # choisir xi distincts aléatoires (évite 0)
        xi_list = self._distinct_x_values(num_shares)
        shares = []
        for xi in xi_list:
            yi = self._evaluate_polynomial(coefficients, xi)
            shares.append((xi, yi))
        return shares

    def reconstruct_secret(self, shares):
        """Reconstruit le secret depuis une liste de (xi, yi)."""
        if len(shares) < self.threshold:
            raise ValueError(f"Besoin d'au moins {self.threshold} shares pour reconstruire")

        # utiliser les threshold premières shares — mais s'assurer d'avoir xi distincts
        # on trie et prend les premières threshold qui ont xi distincts
        unique = []
        seen_x = set()
        for (xi, yi) in shares:
            if xi in seen_x:
                continue
            seen_x.add(xi)
            unique.append((xi, yi))
            if len(unique) == self.threshold:
                break

        if len(unique) < self.threshold:
            raise ValueError("Shares insuffisantes avec xi distincts pour reconstruction")

        secret_int = self._lagrange_interpolation(unique, 0)

        # reconvertir en float
        secret = (secret_int % self.prime) / float(self._scale)
        return secret

    # --------- utilitaires ---------
    def _evaluate_polynomial(self, coefficients, x):
        """Evaluate polynomial coefficients at x modulo prime."""
        res = 0
        for i, a in enumerate(coefficients):
            # pow(x, i, prime) est plus sûr pour grands exposants
            res = (res + (a * pow(x, i, self.prime))) % self.prime
        return res

    def _lagrange_interpolation(self, shares, x):
        """Interpolation de Lagrange mod prime; shares: list[(xi, yi)]."""
        result = 0
        for i, (xi, yi) in enumerate(shares):
            # calcul numerator et denominator pour la base de Lagrange
            num = 1
            den = 1
            for j, (xj, _) in enumerate(shares):
                if i == j:
                    continue
                num = (num * ((x - xj) % self.prime)) % self.prime
                den = (den * ((xi - xj) % self.prime)) % self.prime

            # tenter l'inverse modulaire; si impossible, lever erreur informative
            try:
                inv_den = pow(den, -1, self.prime)  # Python 3.8+: inverse mod; échoue si pas inversible
            except ValueError:
                # cas fréquent: prime non premier ou den divisible par prime
                raise Exception(f"Inverse modulaire inexistant pour den={den} mod {self.prime} — vérifiez que prime est premier et xi distincts")

            term = yi * num * inv_den
            result = (result + term) % self.prime
        return result

    def _distinct_x_values(self, count):
        """Retourne `count` xi distincts dans [1, prime-1]."""
        if count >= self.prime:
            raise ValueError("Trop de shares demandés pour la taille du champ.")
        # Si count petit, on peut tirer au hasard sans risque de collision
        vals = set()
        while len(vals) < count:
            vals.add(random.randrange(1, self.prime))
        return list(vals)

    # --------- primality helpers (probabilistic/simple) ---------
    def _is_probable_prime(self, n):
        """Test de primalité rapide (Miller-Rabin probabiliste)."""
        if n < 2:
            return False
        # petits diviseurs rapides
        small_primes = [2,3,5,7,11,13,17,19,23,29]
        for p in small_primes:
            if n == p:
                return True
            if n % p == 0:
                return False
        # Miller-Rabin avec quelques bases
        d = n - 1
        s = 0
        while d % 2 == 0:
            s += 1
            d //= 2
        # quelques bases communes pour 64-bit (suffisent généralement)
        for a in [2, 325, 9375, 28178, 450775, 9780504, 1795265022]:
            if a % n == 0:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n-1:
                continue
            for _ in range(s-1):
                x = pow(x, 2, n)
                if x == n-1:
                    break
            else:
                return False
        return True

    def _next_prime(self, n):
        """Trouve le prochain nombre premier > n (méthode naïve mais OK pour nombre raisonnable)."""
        candidate = n + 1 if n % 2 == 0 else n + 2
        while True:
            if self._is_probable_prime(candidate):
                return candidate
            candidate += 2
