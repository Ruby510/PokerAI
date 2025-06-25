import streamlit as st
import pandas as pd
import numpy as np
import random
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Poker AI Tutorial & Predictor",
    page_icon="🃏",
    layout="wide"
)

# Constants
SUITS = {'S': '♠', 'H': '♥', 'D': '♦', 'C': '♣'}
RANKS = {'A': 'A', '2': '2', '3': '3', '4': '4', '5': '5', '6': '6', '7': '7', '8': '8', '9': '9', 'T': '10', 'J': 'J', 'Q': 'Q', 'K': 'K'}

# Additional constants for enhanced tutorial/practice
SUITS_NUM = {1: '♥', 2: '♠', 3: '♦', 4: '♣'}
RANKS_NUM = {1: 'A', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 
         9: '9', 10: '10', 11: 'J', 12: 'Q', 13: 'K'}

# Tutorial constants
HAND_NAMES = {
    0: 'High Card', 1: 'One Pair', 2: 'Two Pairs', 3: 'Three of a Kind',
    4: 'Straight', 5: 'Flush', 6: 'Full House', 7: 'Four of a Kind',
    8: 'Straight Flush', 9: 'Royal Flush'
}
HAND_STRENGTH = {
    0: 0.05, 1: 0.15, 2: 0.25, 3: 0.35, 4: 0.50,
    5: 0.60, 6: 0.75, 7: 0.90, 8: 0.95, 9: 0.99
}

# Enhanced helper functions for tutorial/practice
def evaluate_poker_hand(cards):
    """Evaluate poker hand and return class (0-9)"""
    suits = [card[0] for card in cards]
    ranks = sorted([card[1] for card in cards])
    
    # Count rank frequencies
    rank_counts = {}
    for rank in ranks:
        rank_counts[rank] = rank_counts.get(rank, 0) + 1
    
    counts = sorted(rank_counts.values(), reverse=True)
    
    # Check for flush
    is_flush = len(set(suits)) == 1
    
    # Check for straight
    is_straight = False
    if ranks == list(range(ranks[0], ranks[0] + 5)):
        is_straight = True
    elif ranks == [1, 10, 11, 12, 13]:  # A-10-J-Q-K
        is_straight = True
    
    # Classify hand
    if is_straight and is_flush:
        if ranks == [1, 10, 11, 12, 13] or ranks == [9, 10, 11, 12, 13]:
            return 9  # Royal Flush
        else:
            return 8  # Straight Flush
    elif counts[0] == 4:
        return 7  # Four of a Kind
    elif counts[0] == 3 and counts[1] == 2:
        return 6  # Full House
    elif is_flush:
        return 5  # Flush
    elif is_straight:
        return 4  # Straight
    elif counts[0] == 3:
        return 3  # Three of a Kind
    elif counts[0] == 2 and counts[1] == 2:
        return 2  # Two Pairs
    elif counts[0] == 2:
        return 1  # One Pair
    else:
        return 0  # High Card

def predict_action(hand_strength, pot_odds):
    """Simple rule-based action prediction"""
    if hand_strength > 0.7:
        return 2, [0.1, 0.2, 0.7]  # Raise
    elif hand_strength > pot_odds:
        return 1, [0.2, 0.6, 0.2]  # Call
    else:
        return 0, [0.7, 0.2, 0.1]  # Fold

def calculate_win_probability(hand_class, num_opponents=1):
    """Calculate win probability based on hand strength"""
    base_strength = HAND_STRENGTH[hand_class]
    win_prob = base_strength ** num_opponents
    
    if hand_class >= 7:  # Strong hands
        win_prob = max(0.85, win_prob)
    elif hand_class <= 1:  # Weak hands
        win_prob = min(0.25, win_prob)
    
    return win_prob

def generate_explanation(hand_class, cards):
    """Generate educational explanation"""
    hand_name = HAND_NAMES[hand_class]
    card_str = " ".join([f"{RANKS_NUM[rank]}{SUITS_NUM[suit]}" for suit, rank in cards])
    
    explanations = {
        0: f"🃏 **{hand_name}**: You have no pairs. Your highest card matters most.",
        1: f"👥 **{hand_name}**: Two cards of the same rank! Better than high card.",
        2: f"👥👥 **{hand_name}**: Two different pairs! A decent hand.",
        3: f"🎯 **{hand_name}**: Three cards of the same rank! Strong hand.",
        4: f"📈 **{hand_name}**: Five consecutive cards! Very strong.",
        5: f"🌈 **{hand_name}**: Five cards same suit! Strong hand.",
        6: f"🏠 **{hand_name}**: Three of a kind + pair! Excellent!",
        7: f"🔥 **{hand_name}**: Four same rank! Almost unbeatable!",
        8: f"💎 **{hand_name}**: Straight + flush! Nearly unbeatable!",
        9: f"👑 **{hand_name}**: Best possible hand! You cannot lose!"
    }
    
    return explanations.get(hand_class, "Unknown hand")

def generate_action_explanation(action, hand_strength, pot_odds):
    """Generate explanation for recommended action"""
    actions = ["Fold", "Call", "Raise"]
    recommended = actions[action]
    
    if action == 0:  # Fold
        return f"🎯 **Recommended: {recommended}**\n\n💡 Your hand strength ({hand_strength:.1%}) is below pot odds ({pot_odds:.1%}). Save chips for better spots!"
    elif action == 1:  # Call
        return f"🎯 **Recommended: {recommended}**\n\n💡 Your hand strength ({hand_strength:.1%}) justifies the pot odds ({pot_odds:.1%}). Reasonable call."
    else:  # Raise
        return f"🎯 **Recommended: {recommended}**\n\n💡 Strong hand ({hand_strength:.1%})! Bet for value to build the pot."

class RealPokerDataProcessor:
    def __init__(self):
        self.dataset = None
        self.model = None
        self.is_trained = False
        self.training_accuracy = 0.0

    def load_dataset(self, uploaded_file):
        try:
            # Get file extension
            file_name = uploaded_file.name
            file_extension = file_name.split('.')[-1].lower()
            
            # Load based on file type
            if file_extension == 'csv':
                self.dataset = pd.read_csv(uploaded_file)
            elif file_extension == 'txt':
                # Try different delimiters for txt files
                delimiters = ['\t', ',', ' ', '|', ';']
                loaded = False
                
                for delimiter in delimiters:
                    try:
                        uploaded_file.seek(0)  # Reset file pointer
                        self.dataset = pd.read_csv(uploaded_file, delimiter=delimiter)
                        # Check if we got reasonable columns (more than 1 column)
                        if len(self.dataset.columns) > 1:
                            loaded = True
                            st.info(f"✅ Detected delimiter: '{delimiter}'")
                            break
                    except:
                        continue
                
                if not loaded:
                    uploaded_file.seek(0)
                    # Fallback: try as comma-separated
                    self.dataset = pd.read_csv(uploaded_file, delimiter=',')
            else:
                st.error(f"Unsupported file type: {file_extension}")
                return False
            
            # Check for required columns
            required_columns = ['hole_card_1', 'hole_card_2', 'board_card_1', 'board_card_2', 
                              'board_card_3', 'board_card_4', 'board_card_5', 'strength']
            missing_columns = [col for col in required_columns if col not in self.dataset.columns]
            
            if missing_columns:
                st.error(f"❌ Missing required columns: {missing_columns}")
                st.write("**Expected columns:**")
                for col in required_columns:
                    st.write(f"• `{col}`")
                st.write("**Your file has these columns:**")
                for col in self.dataset.columns:
                    st.write(f"• `{col}`")
                st.info("💡 **Tip**: Use 'Use Demo Dataset' to see the expected format, or ensure your file has the correct column names.")
                return False
            
            st.success(f"📁 Successfully loaded {file_extension.upper()} file with {len(self.dataset)} rows")
            return True
            
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
            st.info("💡 **Try**: Make sure your file is properly formatted with the correct columns and delimiters")
            return False

    def create_demo_dataset(self, n_samples=1000):
        demo_data = []
        for _ in range(n_samples):
            hole_cards = self._generate_random_cards(2)
            board_cards = self._generate_random_cards(5, exclude=hole_cards)
            all_cards = hole_cards + board_cards
            
            # Convert to numerical format for proper hand evaluation
            cards_numeric = [(self._suit_to_num(c[1]), self._rank_to_num(c[0])) for c in all_cards]
            hand_class = evaluate_poker_hand(cards_numeric[-5:])  # Use last 5 cards
            hand_strength = HAND_STRENGTH[hand_class] * 100  # Convert to 0-100 scale
            
            row = {
                'hole_card_1': hole_cards[0], 'hole_card_2': hole_cards[1],
                'board_card_1': board_cards[0], 'board_card_2': board_cards[1], 'board_card_3': board_cards[2],
                'board_card_4': board_cards[3], 'board_card_5': board_cards[4], 'strength': hand_strength
            }
            demo_data.append(row)
        self.dataset = pd.DataFrame(demo_data)
        return True
    
    def _suit_to_num(self, suit):
        return {'S': 1, 'H': 2, 'D': 3, 'C': 4}[suit]
    
    def _rank_to_num(self, rank):
        return {'A': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13}[rank]

    def _generate_random_cards(self, n_cards, exclude=None):
        if exclude is None: exclude = []
        all_cards = [f"{r}{s}" for s in SUITS.keys() for r in RANKS.keys()]
        available = [c for c in all_cards if c not in exclude]
        return random.sample(available, n_cards)

    def _evaluate_hand_strength(self, cards):
        ranks = [c[0] for c in cards]
        suits = [c[1] for c in cards]
        counts = sorted([ranks.count(r) for r in set(ranks)], reverse=True)
        if counts[0] >= 4: return random.randint(85, 95)
        if counts[0] >= 3 and counts[1] >= 2: return random.randint(80, 90)
        if max([suits.count(s) for s in set(suits)]) >= 5: return random.randint(70, 85)
        if counts[0] >= 3: return random.randint(60, 75)
        if counts[0] >= 2 and counts[1] >= 2: return random.randint(45, 65)
        if counts[0] >= 2: return random.randint(25, 50)
        return random.randint(10, 35)

    def prepare_features(self):
        rank_map = {'A': 14, 'K': 13, 'Q': 12, 'J': 11, 'T': 10, '9': 9, '8': 8, '7': 7, '6': 6, '5': 5, '4': 4, '3': 3, '2': 2}
        suit_map = {'S': 1, 'H': 2, 'D': 3, 'C': 4}
        features, targets = [], []
        
        for _, row in self.dataset.iterrows():
            hole = [row['hole_card_1'], row['hole_card_2']]
            board = [row[f'board_card_{i}'] for i in range(1, 6)]
            cards = hole + board
            
            # Basic features: rank and suit
            feature = []
            for c in cards:
                feature.extend([rank_map[c[0]], suit_map[c[1]]])
            
            # Enhanced poker features
            ranks = [rank_map[c[0]] for c in cards]
            suits = [suit_map[c[1]] for c in cards]
            
            # Add poker-specific features
            rank_counts = [ranks.count(r) for r in set(ranks)]
            suit_counts = [suits.count(s) for s in set(suits)]
            
            # Feature engineering
            feature.extend([
                max(rank_counts),  # Highest pair/trips/quads
                len([c for c in rank_counts if c >= 2]),  # Number of pairs
                max(suit_counts),  # Flush potential
                max(ranks),  # Highest card
                min(ranks),  # Lowest card
                len(set(ranks)),  # Number of unique ranks
            ])
            
            features.append(feature)
            targets.append(row['strength'] / 100)
            
        return np.array(features), np.array(targets)

    def train_model(self):
        try:
            X, y = self.prepare_features()
            if len(X) == 0: 
                st.error("No valid data found for training.")
                return False
            X_bias = np.column_stack([np.ones(len(X)), X])
            self.model = np.linalg.lstsq(X_bias, y, rcond=None)[0]
            preds = np.clip(X_bias @ self.model, 0, 1)
            self.training_accuracy = 1 - np.sum((y - preds)**2) / np.sum((y - y.mean())**2)
            self.is_trained = True
            return True
        except Exception as e:
            st.error(f"Error training model: {e}")
            return False

    def predict(self, hole, board):
        if not self.is_trained: 
            return 0.5
        
        rank_map = {'A': 14, 'K': 13, 'Q': 12, 'J': 11, 'T': 10, '9': 9, '8': 8, '7': 7, '6': 6, '5': 5, '4': 4, '3': 3, '2': 2}
        suit_map = {'S': 1, 'H': 2, 'D': 3, 'C': 4}
        
        cards = hole + board
        
        # Basic features: rank and suit for each card
        features = []
        for c in cards:
            features.extend([rank_map.get(c[0], 2), suit_map.get(c[1], 1)])
        
        # Enhanced poker features (same as in prepare_features)
        ranks = [rank_map.get(c[0], 2) for c in cards]
        suits = [suit_map.get(c[1], 1) for c in cards]
        
        # Add poker-specific features
        rank_counts = [ranks.count(r) for r in set(ranks)]
        suit_counts = [suits.count(s) for s in set(suits)]
        
        # Feature engineering (same as training)
        features.extend([
            max(rank_counts),  # Highest pair/trips/quads
            len([c for c in rank_counts if c >= 2]),  # Number of pairs
            max(suit_counts),  # Flush potential
            max(ranks),  # Highest card
            min(ranks),  # Lowest card
            len(set(ranks)),  # Number of unique ranks
        ])
        
        # Add bias term
        X = np.hstack(([1], features))
        
        # Debug: Check dimensions
        if len(X) != len(self.model):
            st.error(f"❌ Feature mismatch! Expected {len(self.model)} features, got {len(X)}. Please retrain the model.")
            st.info("Click 'Use Demo Dataset' or reload your dataset to retrain the model with correct features.")
            return 0.5
        
        pred = np.dot(X, self.model)
        return np.clip(pred, 0, 1)

def parse_card(card_str):
    return f"{RANKS.get(card_str[0], card_str[0])}{SUITS.get(card_str[1], card_str[1])}" if len(card_str) == 2 else card_str

if 'processor' not in st.session_state:
    st.session_state.processor = RealPokerDataProcessor()
if 'hands_analyzed' not in st.session_state:
    st.session_state.hands_analyzed = 0
    st.session_state.correct_predictions = 0

st.title("🃏 Poker AI Tutorial & Predictor")

with st.sidebar:
    uploaded_file = st.file_uploader("Upload Dataset (CSV/TXT)", type=['csv', 'txt'])
    
    if uploaded_file and st.button("Load Dataset"):
        if st.session_state.processor.load_dataset(uploaded_file):
            if st.session_state.processor.train_model():
                st.success(f"Dataset loaded and model trained! Accuracy: {st.session_state.processor.training_accuracy:.1%}")
                st.info("✅ Model trained with correct features!")
            else:
                st.error("Failed to train model. Please check your dataset format.")
    if st.button("Use Demo Dataset"):
        st.session_state.processor.create_demo_dataset()
        st.session_state.processor.train_model()
        st.success(f"Demo dataset created and model trained! Accuracy: {st.session_state.processor.training_accuracy:.1%}")
        st.info("✅ Model retrained with correct features!")
    st.metric("Hands Analyzed", st.session_state.hands_analyzed)
    accuracy = (st.session_state.correct_predictions / max(1, st.session_state.hands_analyzed)) * 100
    
    # Model status indicator
    if st.session_state.processor.is_trained:
        st.success("✅ Model Ready")
        st.metric("Model Accuracy", f"{st.session_state.processor.training_accuracy:.1%}")
    else:
        st.warning("⚠️ Model Not Trained")
    
    if st.button("🔄 Reset Progress"):
        st.session_state.hands_analyzed = 0
        st.session_state.correct_predictions = 0
        # Also reset the model to fix any feature mismatch issues
        st.session_state.processor = RealPokerDataProcessor()
        if 'result' in st.session_state:
            del st.session_state.result
        if 'quiz_hand' in st.session_state:
            del st.session_state.quiz_hand
        if 'practice_scenario' in st.session_state:
            del st.session_state.practice_scenario
        st.rerun()

if st.session_state.processor.dataset is not None:
    tabs = st.tabs(["Analyzer", "Dataset", "Tutorial", "Practice"])

    with tabs[0]:
        st.header("Analyze Poker Hand")
        
        if not st.session_state.processor.is_trained:
            st.warning("⚠️ Model not trained yet! Please load a dataset or use demo dataset first.")
            st.info("👈 Use the sidebar to either upload a CSV file or click 'Use Demo Dataset' to start analyzing hands.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.write("Hole Cards:")
                hole1_suit = st.selectbox("Hole Card 1 Suit", options=list(SUITS.keys()), format_func=lambda s: SUITS[s], key="hole1_suit")
                hole1_rank = st.selectbox("Hole Card 1 Rank", options=list(RANKS.keys()), format_func=lambda r: RANKS[r], key="hole1_rank")
                hole2_suit = st.selectbox("Hole Card 2 Suit", options=list(SUITS.keys()), format_func=lambda s: SUITS[s], key="hole2_suit")
                hole2_rank = st.selectbox("Hole Card 2 Rank", options=list(RANKS.keys()), format_func=lambda r: RANKS[r], key="hole2_rank")
                hole1 = f"{hole1_rank}{hole1_suit}"
                hole2 = f"{hole2_rank}{hole2_suit}"
                st.write("Board Cards:")
                board = []
                for i in range(5):
                    suit = st.selectbox(f"Board Card {i+1} Suit", options=list(SUITS.keys()), format_func=lambda s: SUITS[s], key=f"board_suit_{i}")
                    rank = st.selectbox(f"Board Card {i+1} Rank", options=list(RANKS.keys()), format_func=lambda r: RANKS[r], key=f"board_rank_{i}")
                    board.append(f"{rank}{suit}")
                if st.button("Analyze Hand"):
                    win_prob = st.session_state.processor.predict([hole1, hole2], board)
                    st.session_state.result = (hole1, hole2, board, win_prob)
            
            with col2:
                if 'result' in st.session_state:
                    hole1, hole2, board, win_prob = st.session_state.result
                    st.write(f"Your Hand: {parse_card(hole1)} {parse_card(hole2)}")
                    st.write(f"Board: {' '.join([parse_card(c) for c in board if c])}")
                    st.metric("Win Probability", f"{win_prob:.1%}")
                    st.progress(win_prob, text=f"{win_prob:.1%}")
                    st.session_state.hands_analyzed += 1

    with tabs[1]:  
        st.write(st.session_state.processor.dataset.head())
        st.write("Total Hands:", len(st.session_state.processor.dataset))
        if 'strength' in st.session_state.processor.dataset.columns:
            fig = px.histogram(st.session_state.processor.dataset, x='strength', nbins=20, title='Hand Strength Distribution')
            st.plotly_chart(fig)

    with tabs[2]:
        st.header("📚 Poker Tutorial")
        
        lesson = st.selectbox(
            "Choose Lesson",
            ["Hand Rankings", "Pot Odds", "Position Strategy", "Betting Basics"]
        )
        
        if lesson == "Hand Rankings":
            st.subheader("🃏 Poker Hand Rankings (Strongest to Weakest)")
            
            for rank in range(9, -1, -1):
                strength = HAND_STRENGTH[rank]
                st.write(f"**{10-rank}. {HAND_NAMES[rank]}** - {strength:.0%} strength")
            
            # Interactive quiz
            st.write("---")
            st.subheader("🎯 Hand Recognition Quiz")
            
            if st.button("Generate Quiz Hand"):
                quiz_cards = []
                used_cards = set()
                while len(quiz_cards) < 5:
                    suit = random.randint(1, 4)
                    rank = random.randint(1, 13)
                    card = (suit, rank)
                    if card not in used_cards:
                        quiz_cards.append(card)
                        used_cards.add(card)
                
                st.session_state.quiz_hand = quiz_cards
                st.session_state.quiz_answer = evaluate_poker_hand(quiz_cards)
            
            if 'quiz_hand' in st.session_state:
                quiz_display = " ".join([f"{RANKS_NUM[rank]}{SUITS_NUM[suit]}" for suit, rank in st.session_state.quiz_hand])
                st.write(f"**Quiz Hand:** {quiz_display}")
                
                user_guess = st.selectbox("What hand is this?", list(HAND_NAMES.values()))
                
                if st.button("Check Answer"):
                    correct_answer = HAND_NAMES[st.session_state.quiz_answer]
                    if user_guess == correct_answer:
                        st.success("✅ Correct!")
                        st.session_state.correct_predictions += 1
                    else:
                        st.error(f"❌ Wrong. Correct answer: {correct_answer}")
                    
                    st.session_state.hands_analyzed += 1
                    
                    explanation = generate_explanation(st.session_state.quiz_answer, st.session_state.quiz_hand)
                    st.info(explanation)
        
        elif lesson == "Pot Odds":
            st.subheader("🧮 Understanding Pot Odds")
            
            st.write("""
            **Pot Odds** help you decide if a call is profitable:
            
            📊 **Formula:** Pot Odds = Bet to Call ÷ (Total Pot + Bet to Call)
            
            🎯 **Rule:** Call when your winning chances > Pot Odds percentage
            """)
            
            # Calculator
            st.write("**Calculator:**")
            calc_pot = st.number_input("Current Pot", min_value=1, value=100)
            calc_bet = st.number_input("Bet to Call", min_value=1, value=25)
            
            calc_odds = calc_bet / (calc_pot + calc_bet)
            st.metric("Pot Odds", f"{calc_odds:.1%}")
            st.write(f"You need at least {calc_odds:.1%} chance of winning to call profitably")
        
        else:
            st.info(f"📚 {lesson} lesson coming soon!")

    with tabs[3]:
        st.header("🎲 Practice Mode")
        
        st.write("Test your poker skills!")
        
        if st.button("🎯 New Practice Hand"):
            practice_cards = []
            used_cards = set()
            while len(practice_cards) < 5:
                suit = random.randint(1, 4)
                rank = random.randint(1, 13)
                card = (suit, rank)
                if card not in used_cards:
                    practice_cards.append(card)
                    used_cards.add(card)
            
            practice_pot = random.randint(20, 200)
            practice_bet = random.randint(5, practice_pot // 2)
            
            st.session_state.practice_scenario = {
                'cards': practice_cards,
                'pot': practice_pot,
                'bet': practice_bet,
                'hand_class': evaluate_poker_hand(practice_cards)
            }
        
        if 'practice_scenario' in st.session_state:
            scenario = st.session_state.practice_scenario
            
            # Display scenario
            card_display = " ".join([f"{RANKS_NUM[rank]}{SUITS_NUM[suit]}" for suit, rank in scenario['cards']])
            st.write(f"**Hand:** {card_display}")
            st.write(f"**Pot:** ${scenario['pot']} | **Bet to Call:** ${scenario['bet']}")
            
            # Get user predictions
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**What hand do you have?**")
                hand_guess = st.selectbox("Hand Type", list(HAND_NAMES.values()), key="hand_guess")
            
            with col2:
                st.write("**What should you do?**")
                action_guess = st.selectbox("Action", ["Fold", "Call", "Raise"], key="action_guess")
            
            if st.button("Submit Answers"):
                # Check hand recognition
                correct_hand = HAND_NAMES[scenario['hand_class']]
                hand_correct = hand_guess == correct_hand
                
                # Check action
                hand_strength = HAND_STRENGTH[scenario['hand_class']]
                pot_odds = scenario['bet'] / (scenario['pot'] + scenario['bet'])
                optimal_action, _ = predict_action(hand_strength, pot_odds)
                action_names = ["Fold", "Call", "Raise"]
                correct_action = action_names[optimal_action]
                action_correct = action_guess == correct_action
                
                # Update scoring
                if hand_correct:
                    st.session_state.correct_predictions += 1
                if action_correct:
                    st.session_state.correct_predictions += 1
                
                st.session_state.hands_analyzed += 1
                
                # Show results
                if hand_correct:
                    st.success(f"✅ Hand: Correct! It's {correct_hand}")
                else:
                    st.error(f"❌ Hand: Wrong. It's {correct_hand}")
                
                if action_correct:
                    st.success(f"✅ Action: Correct! {correct_action} is optimal")
                else:
                    st.error(f"❌ Action: Suboptimal. Better choice: {correct_action}")
                
                # Show explanations
                explanation = generate_explanation(scenario['hand_class'], scenario['cards'])
                st.info(explanation)
                
                action_explanation = generate_action_explanation(optimal_action, hand_strength, pot_odds)
                st.info(action_explanation)









