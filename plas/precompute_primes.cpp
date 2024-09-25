#include<fstream>
#include<iostream>
#include<vector>
using namespace std;

const int N = 1'000'000'000;

int main(int argc, char const *argv[])
{
    vector<int> primes;
    vector<int> lowest_prime_factor(N+1);

    for (int i=2; i <= N; ++i) {
        if (lowest_prime_factor[i] == 0) {
            lowest_prime_factor[i] = i;
            primes.push_back(i);
        }
        for (int j = 0; i * primes[j] <= N; ++j) {
            lowest_prime_factor[i * primes[j]] = primes[j];
            if (primes[j] == lowest_prime_factor[i]) {
                break;
            }
        }
    }

    ofstream out("primes.txt");
    if (out.is_open()) {
        out << primes.size() << endl;
        for (int p : primes) {
            out << p << endl;
        }
        out.close();
    }
    return 0;
}
