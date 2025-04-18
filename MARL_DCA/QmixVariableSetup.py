import argparse

# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="QMIX_CTDE Training Parameters")

    parser.add_argument("--n", type=int, default=5,             help="Number of agents")
    parser.add_argument("--epsilon", type=float, default=0.1,   help="Epsilon for epsilon-greedy policy")
    parser.add_argument("--gamma", type=float, default=0.99,    help="Discount factor")
    parser.add_argument("--T", type=int, default=1000,          help="Total training steps")
    parser.add_argument("--Nr", type=int, default=10,           help="Target network update frequency")
    parser.add_argument("--batch_size", type=int, default=32,   help="Batch size for training")
    parser.add_argument("--memory_size", type=int, default=10000, help="Experience replay memory size")
    parser.add_argument("--lr", type=float, default=0.001,      help="Learning rate for optimizers")

    parser.add_argument("--t", type=int, default=0,             help="Current time step")
    parser.add_argument("--cnt", type=int, default=0,           help="Counter for updates")
    parser.add_argument("--s0", type=float, default=0.0,        help="Initial global state")
    parser.add_argument("--tau0", type=float, default=0.0,      help="Initial local observation for each agent")
    parser.add_argument("--a0", type=str, default="Wait",       help="Initial action for each agent")
    parser.add_argument("--z0", type=int, default=0,            help="Initial transmission status for each agent")
    parser.add_argument("--beta0", type=float, default=1.0,     help="Initial weighting factor for each agent")
    parser.add_argument("--theta0", type=float, default=0.0,    help="Initial neural network parameters")
    parser.add_argument("--theta_target0", type=float, default=0.0, help="Initial target network parameters")
    
    return parser.parse_args()