import streamlit as st
import pandas as pd
import numpy as np
import random
import plotly.express as px
from datetime import datetime

st.set_page_config(
    page_title="Poker AI Tutorial & Predictor",
    page_icon="🃏",
    layout="wide"
)

# Simple card definitions
SUITS = {'S': '♠', 'H': '♥', 'D': '♦', 'C': '♣'}
RANKS = {'A': 'A', '2': '2', '3': '3', '4': '4', '5': '5', '6': '6', '7': '7', 
         '8': '8', '9': '9', 'T': '10', 'J': 'J', 'Q': 'Q', 'K': 'K'}

# Card display symbols
SUIT_SYMBOLS = {1: '♥', 2: '♠', 3: '♦', 4: '♣'}
RANK_NAMES = {1: 'A', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 
              9: '9', 10: '10', 11: 'J', 12: 'Q', 13: 'K'}

# Poker hand types and strengths
HAND_TYPES = {
    0: 'High Card', 1: 'One Pair', 2: 'Two Pairs', 3: 'Three of a Kind',
    4: 'Straight', 5: 'Flush', 6: 'Full House', 7: 'Four of a Kind',
    8: 'Straight Flush', 9: 'Royal Flush'
}

HAND_STRENGTHS = {
    0: 0.05, 1: 0.15, 2: 0.25, 3: 0.35, 4: 0.50,
    5: 0.60, 6: 0.75, 7: 0.90, 8: 0.95, 9: 0.99
}

def check_poker_hand(cards):
    suits = [card[0] for card in cards] 
    ranks = sorted([card[1] for card in cards])  
    
    # Count rank
    rank_counts = {}
    for rank in ranks:
        if rank in rank_counts:
            rank_counts[rank] += 1
        else:
            rank_counts[rank] = 1
    
    # Sort the counts
    counts = sorted(rank_counts.values(), reverse=True)
    
    # Check if all cards are same suit (flush)
    all_same_suit = len(set(suits)) == 1
    
    # Check if cards are in sequence (straight)
    is_straight = False
    if ranks == list(range(ranks[0], ranks[0] + 5)):
        is_straight = True
    elif ranks == [1, 10, 11, 12, 13]:  # Special case: A-10-J-Q-K
        is_straight = True
    
    # Figure the hand
    if is_straight and all_same_suit:
        if ranks == [1, 10, 11, 12, 13] or ranks == [9, 10, 11, 12, 13]:
            return 9  # Royal Flush - best hand!
        else:
            return 8  # Straight Flush
    elif counts[0] == 4:
        return 7  # Four of a Kind
    elif counts[0] == 3 and counts[1] == 2:
        return 6  # Full House
    elif all_same_suit:
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

def decide_action(hand_strength, pot_odds):
    """Decide whether to fold, call, or raise"""
    if hand_strength > 0.7:
        return 2, [0.1, 0.2, 0.7]  # Raise
    elif hand_strength > pot_odds:
        return 1, [0.2, 0.6, 0.2]  # Call
    else:
        return 0, [0.7, 0.2, 0.1]  # Fold

# Calculate winning chances
def calculate_win_chance(hand_type, opponents=1):
    base_chance = HAND_STRENGTHS[hand_type]
    win_chance = base_chance ** opponents
    
    if hand_type >= 7:  # Very strong hands
        win_chance = max(0.85, win_chance)
    elif hand_type <= 1:  # Weak hands
        win_chance = min(0.25, win_chance)
    
    return win_chance

# Create explanations for different hands
def explain_hand(hand_type, cards):
    hand_name = HAND_TYPES[hand_type]
    
    explanations = {
        0: f"🃏 **{hand_name}**: No pairs - your highest card matters most.",
        1: f"👥 **{hand_name}**: Two cards of the same rank! Better than high card.",
        2: f"👥👥 **{hand_name}**: Two different pairs! A decent hand.",
        3: f"🎯 **{hand_name}**: Three cards of the same rank! Strong hand.",
        4: f"📈 **{hand_name}**: Five cards in a row! Very strong.",
        5: f"🌈 **{hand_name}**: Five cards of the same suit! Strong hand.",
        6: f"🏠 **{hand_name}**: Three of a kind + pair! Excellent!",
        7: f"🔥 **{hand_name}**: Four cards of the same rank! Almost unbeatable!",
        8: f"💎 **{hand_name}**: Straight + flush combo! Nearly unbeatable!",
        9: f"👑 **{hand_name}**: The best possible hand! You cannot lose!"
    }
    
    return explanations.get(hand_type, "Unknown hand")

# Explain action
def explain_action(action, hand_strength, pot_odds):
    """Explain why we should fold, call, or raise"""
    actions = ["Fold", "Call", "Raise"]
    recommended = actions[action]
    
    if action == 0:  # Fold
        return f"🎯 **Recommended: {recommended}**\n\n💡 Your hand strength ({hand_strength:.1%}) is weaker than the pot odds ({pot_odds:.1%}). Better to save your chips!"
    elif action == 1:  # Call
        return f"🎯 **Recommended: {recommended}**\n\n💡 Your hand strength ({hand_strength:.1%}) is good enough for the pot odds ({pot_odds:.1%}). A reasonable call."
    else:  # Raise
        return f"🎯 **Recommended: {recommended}**\n\n💡 Strong hand ({hand_strength:.1%})! Bet to build the pot and win more money."

class SimplePokerAI:
    def __init__(self):
        self.data = None
        self.model_weights = None
        self.is_ready = False
        self.accuracy = 0.0

    def load_data_file(self, file):
        try:
            # Check what kind of file it is
            filename = file.name
            file_type = filename.split('.')[-1].lower()
            
            # Read the file
            if file_type == 'csv':
                self.data = pd.read_csv(file)
            elif file_type == 'txt':
                # Try different separators for text files
                separators = ['\t', ',', ' ', '|', ';']
                loaded = False
                
                for sep in separators:
                    try:
                        file.seek(0)  # Go back to start of file
                        self.data = pd.read_csv(file, delimiter=sep)
                        if len(self.data.columns) > 1: 
                            loaded = True
                            st.info(f"✅ Found separator: '{sep}'")
                            break
                    except:
                        continue
                
                if not loaded:
                    file.seek(0)
                    self.data = pd.read_csv(file, delimiter=',')
            else:
                st.error(f"Can't read {file_type} files")
                return False
            
            # Check for the right columns
            needed_columns = ['hole_card_1', 'hole_card_2', 'board_card_1', 'board_card_2', 
                             'board_card_3', 'board_card_4', 'board_card_5', 'strength']
            missing = [col for col in needed_columns if col not in self.data.columns]
            
            if missing:
                st.error(f"❌ Missing columns: {missing}")
                st.write("**Your file needs these columns:**")
                for col in needed_columns:
                    st.write(f"• `{col}`")
                st.write("**Your file has:**")
                for col in self.data.columns:
                    st.write(f"• `{col}`")
                return False
            
            st.success(f"📁 Loaded {len(self.data)} poker hands from {file_type.upper()} file")
            return True
            
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return False

    def create_sample_data(self, num_hands=1000):
        sample_data = []
        
        for _ in range(num_hands):
            # Generate random cards
            hole_cards = self.make_random_cards(2)
            board_cards = self.make_random_cards(5, exclude=hole_cards)
            all_cards = hole_cards + board_cards
            
            # Convert to numbers
            cards_as_numbers = [(self.suit_to_number(c[1]), self.rank_to_number(c[0])) for c in all_cards]
            hand_type = check_poker_hand(cards_as_numbers[-5:])  # Use last 5 cards
            strength = HAND_STRENGTHS[hand_type] * 100  # Convert to 0-100 scale
            
            # Create a row of data
            row = {
                'hole_card_1': hole_cards[0], 'hole_card_2': hole_cards[1],
                'board_card_1': board_cards[0], 'board_card_2': board_cards[1], 
                'board_card_3': board_cards[2], 'board_card_4': board_cards[3], 
                'board_card_5': board_cards[4], 'strength': strength
            }
            sample_data.append(row)
        
        self.data = pd.DataFrame(sample_data)
        return True
    
    def suit_to_number(self, suit):
        return {'S': 1, 'H': 2, 'D': 3, 'C': 4}[suit]
    
    def rank_to_number(self, rank):
        return {'A': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, 
                '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13}[rank]

    def make_random_cards(self, num_cards, exclude=None):
        if exclude is None: 
            exclude = []
        
        # All possible cards
        all_cards = [f"{r}{s}" for s in SUITS.keys() for r in RANKS.keys()]
        # Remove cards
        available = [c for c in all_cards if c not in exclude]
        return random.sample(available, num_cards)

    def prepare_training_data(self):
        # Convert ranks and suits to numbers
        rank_values = {'A': 14, 'K': 13, 'Q': 12, 'J': 11, 'T': 10, '9': 9, '8': 8, 
                      '7': 7, '6': 6, '5': 5, '4': 4, '3': 3, '2': 2}
        suit_values = {'S': 1, 'H': 2, 'D': 3, 'C': 4}
        
        features = []
        targets = []
        
        for _, row in self.data.iterrows():
            # Get all cards
            hole = [row['hole_card_1'], row['hole_card_2']]
            board = [row[f'board_card_{i}'] for i in range(1, 6)]
            all_cards = hole + board
            
            # Convert each card to numbers
            card_features = []
            for card in all_cards:
                card_features.extend([rank_values[card[0]], suit_values[card[1]]])
            
            # Add poker-specific features
            ranks = [rank_values[card[0]] for card in all_cards]
            suits = [suit_values[card[1]] for card in all_cards]
            
            # Count pairs, flushes and others
            rank_counts = [ranks.count(r) for r in set(ranks)]
            suit_counts = [suits.count(s) for s in set(suits)]
            
            card_features.extend([
                max(rank_counts),  # Best pair/trips/quads
                len([c for c in rank_counts if c >= 2]),  # Number of pairs
                max(suit_counts),  # Flush potential
                max(ranks),  # Highest card
                min(ranks),  # Lowest card
                len(set(ranks)),  # Number of different ranks
            ])
            
            features.append(card_features)
            targets.append(row['strength'] / 100)
            
        return np.array(features), np.array(targets)

    def train_simple_model(self):
        try:
            X, y = self.prepare_training_data()
            if len(X) == 0: 
                st.error("No data to train with")
                return False
            
            # Add bias term
            X_with_bias = np.column_stack([np.ones(len(X)), X])
            
            # Simple linear regression
            self.model_weights = np.linalg.lstsq(X_with_bias, y, rcond=None)[0]
            
            # Calculate the model
            predictions = np.clip(X_with_bias @ self.model_weights, 0, 1)
            self.accuracy = 1 - np.sum((y - predictions)**2) / np.sum((y - y.mean())**2)
            self.is_ready = True
            return True
            
        except Exception as e:
            st.error(f"Training failed: {e}")
            return False

    def predict_hand_strength(self, hole_cards, board_cards):
        if not self.is_ready: 
            return 0.5  
        
        rank_values = {'A': 14, 'K': 13, 'Q': 12, 'J': 11, 'T': 10, '9': 9, '8': 8, 
                      '7': 7, '6': 6, '5': 5, '4': 4, '3': 3, '2': 2}
        suit_values = {'S': 1, 'H': 2, 'D': 3, 'C': 4}
        
        all_cards = hole_cards + board_cards
        
        # Convert cards to features
        features = []
        for card in all_cards:
            features.extend([rank_values.get(card[0], 2), suit_values.get(card[1], 1)])
        
        # Add the same poker features as training
        ranks = [rank_values.get(card[0], 2) for card in all_cards]
        suits = [suit_values.get(card[1], 1) for card in all_cards]
        
        rank_counts = [ranks.count(r) for r in set(ranks)]
        suit_counts = [suits.count(s) for s in set(suits)]
        
        features.extend([
            max(rank_counts),
            len([c for c in rank_counts if c >= 2]),
            max(suit_counts),
            max(ranks),
            min(ranks),
            len(set(ranks)),
        ])
        
        # Add bias and predict
        X = np.hstack(([1], features))
        
        # Check if dimensions match
        if len(X) != len(self.model_weights):
            st.error(f"❌ Model mismatch! Expected {len(self.model_weights)} features, got {len(X)}. Please retrain.")
            return 0.5
        
        prediction = np.dot(X, self.model_weights)
        return np.clip(prediction, 0, 1)

# Helper function to display cards
def display_card(card_string):
    if len(card_string) == 2:
        return f"{RANKS.get(card_string[0], card_string[0])}{SUITS.get(card_string[1], card_string[1])}"
    return card_string

# Initialize app state
if 'poker_ai' not in st.session_state:
    st.session_state.poker_ai = SimplePokerAI()
if 'hands_played' not in st.session_state:
    st.session_state.hands_played = 0
    st.session_state.correct_guesses = 0

# Main app title
st.title("🃏 Poker AI Tutorial & Predictor")

# Sidebar for data and model management
with st.sidebar:
    st.header("📊 Data & Model")
    
    # File upload
    uploaded_file = st.file_uploader("Upload Poker Data (CSV/TXT)", type=['csv', 'txt'])
    
    if uploaded_file and st.button("📁 Load Data"):
        if st.session_state.poker_ai.load_data_file(uploaded_file):
            if st.session_state.poker_ai.train_simple_model():
                st.success(f"✅ Model trained! Accuracy: {st.session_state.poker_ai.accuracy:.1%}")
            else:
                st.error("❌ Training failed")
    
    # Demo data button
    if st.button("🎲 Use Demo Data"):
        st.session_state.poker_ai.create_sample_data()
        st.session_state.poker_ai.train_simple_model()
        st.success(f"✅ Demo data created and model trained! Accuracy: {st.session_state.poker_ai.accuracy:.1%}")
    
    # Stats
    st.metric("Hands Analyzed", st.session_state.hands_played)
    if st.session_state.hands_played > 0:
        accuracy = (st.session_state.correct_guesses / st.session_state.hands_played) * 100
        st.metric("Your Accuracy", f"{accuracy:.1f}%")
    
    # Model status
    if st.session_state.poker_ai.is_ready:
        st.success("✅ AI Model Ready")
        st.metric("Model Accuracy", f"{st.session_state.poker_ai.accuracy:.1%}")
    else:
        st.warning("⚠️ Load data first")
    
    # Reset button
    if st.button("🔄 Reset Everything"):
        st.session_state.hands_played = 0
        st.session_state.correct_guesses = 0
        st.session_state.poker_ai = SimplePokerAI()
        # Clear any stored results
        for key in list(st.session_state.keys()):
            if key.startswith(('result', 'quiz', 'practice')):
                del st.session_state[key]
        st.rerun()

# Main content area
if st.session_state.poker_ai.data is not None:
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Hand Analyzer", "📊 Data View", "📚 Learn Poker", "🎮 Practice"])

    # Tab 1: Hand Analyzer
    with tab1:
        st.header("🎯 Analyze Your Poker Hand")
        
        if not st.session_state.poker_ai.is_ready:
            st.warning("⚠️ No AI model loaded! Please load data or use demo data first.")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Your Cards")
                
                # Hole cards
                st.write("**Hole Cards (your private cards):**")
                hole1_suit = st.selectbox("Card 1 Suit", options=list(SUITS.keys()), 
                                        format_func=lambda s: SUITS[s], key="h1_suit")
                hole1_rank = st.selectbox("Card 1 Rank", options=list(RANKS.keys()), 
                                        format_func=lambda r: RANKS[r], key="h1_rank")
                hole2_suit = st.selectbox("Card 2 Suit", options=list(SUITS.keys()), 
                                        format_func=lambda s: SUITS[s], key="h2_suit")
                hole2_rank = st.selectbox("Card 2 Rank", options=list(RANKS.keys()), 
                                        format_func=lambda r: RANKS[r], key="h2_rank")
                
                hole1 = f"{hole1_rank}{hole1_suit}"
                hole2 = f"{hole2_rank}{hole2_suit}"
                
                # Board cards
                st.write("**Board Cards (community cards):**")
                board_cards = []
                for i in range(5):
                    suit = st.selectbox(f"Board {i+1} Suit", options=list(SUITS.keys()), 
                                      format_func=lambda s: SUITS[s], key=f"b{i}_suit")
                    rank = st.selectbox(f"Board {i+1} Rank", options=list(RANKS.keys()), 
                                      format_func=lambda r: RANKS[r], key=f"b{i}_rank")
                    board_cards.append(f"{rank}{suit}")
                
                if st.button("🔍 Analyze Hand", type="primary"):
                    strength = st.session_state.poker_ai.predict_hand_strength([hole1, hole2], board_cards)
                    st.session_state.analysis_result = (hole1, hole2, board_cards, strength)
            
            with col2:
                st.subheader("Analysis Results")
                
                if 'analysis_result' in st.session_state:
                    hole1, hole2, board_cards, strength = st.session_state.analysis_result
                    
                    st.write(f"**Your Hand:** {display_card(hole1)} {display_card(hole2)}")
                    st.write(f"**Board:** {' '.join([display_card(c) for c in board_cards])}")
                    
                    # Show strength
                    st.metric("Hand Strength", f"{strength:.1%}")
                    st.progress(strength, text=f"Strength: {strength:.1%}")
                    
                    # Update counter
                    st.session_state.hands_played += 1
                    
                    # Show recommendation
                    if strength > 0.7:
                        st.success("💪 Strong hand! Consider betting/raising")
                    elif strength > 0.4:
                        st.info("👍 Decent hand. Play carefully")
                    else:
                        st.warning("😬 Weak hand. Consider folding")

    # Tab 2: Data View
    with tab2:
        st.header("📊 Poker Data Overview")
        
        st.write("**First few rows of data:**")
        st.dataframe(st.session_state.poker_ai.data.head(10))
        
        st.metric("Total Hands in Dataset", len(st.session_state.poker_ai.data))
        
        if 'strength' in st.session_state.poker_ai.data.columns:
            st.write("**Hand Strength Distribution:**")
            fig = px.histogram(st.session_state.poker_ai.data, x='strength', nbins=20, 
                             title='How Strong Are the Hands in Our Data?')
            fig.update_layout(xaxis_title="Hand Strength", yaxis_title="Number of Hands")
            st.plotly_chart(fig, use_container_width=True)

    # Tab 3: Tutorial
    with tab3:
        st.header("📚 Learn Poker")
        
        lesson_choice = st.selectbox(
            "Choose a lesson:",
            ["🏆 Hand Rankings", "🧮 Pot Odds", "📍 Position Play", "💰 Betting Strategy"]
        )
        
        if lesson_choice == "🏆 Hand Rankings":
            st.subheader("🏆 Poker Hand Rankings (Best to Worst)")
            
            st.write("Here are all poker hands from strongest to weakest:")
            
            for rank in range(9, -1, -1):
                strength = HAND_STRENGTHS[rank]
                position = 10 - rank
                st.write(f"**{position}. {HAND_TYPES[rank]}** - {strength:.0%} strength")
            
            st.write("---")
            st.subheader("🧠 Test Your Knowledge")
            
            if st.button("🎲 Generate Random Hand"):
                # Create a random 5-card hand
                random_hand = []
                used_cards = set()
                
                while len(random_hand) < 5:
                    suit = random.randint(1, 4)
                    rank = random.randint(1, 13)
                    card = (suit, rank)
                    if card not in used_cards:
                        random_hand.append(card)
                        used_cards.add(card)
                
                st.session_state.quiz_hand = random_hand
                st.session_state.quiz_answer = check_poker_hand(random_hand)
            
            if 'quiz_hand' in st.session_state:
                # Show the hand
                hand_display = " ".join([f"{RANK_NAMES[rank]}{SUIT_SYMBOLS[suit]}" 
                                       for suit, rank in st.session_state.quiz_hand])
                st.write(f"**What hand is this?** {hand_display}")
                
                # Let user guess
                user_guess = st.selectbox("Your answer:", list(HAND_TYPES.values()))
                
                if st.button("✅ Check Answer"):
                    correct = HAND_TYPES[st.session_state.quiz_answer]
                    if user_guess == correct:
                        st.success("🎉 Correct!")
                        st.session_state.correct_guesses += 1
                    else:
                        st.error(f"❌ Wrong. The correct answer is: {correct}")
                    
                    st.session_state.hands_played += 1
                    
                    # Show explanation
                    explanation = explain_hand(st.session_state.quiz_answer, st.session_state.quiz_hand)
                    st.info(explanation)
        
        elif lesson_choice == "🧮 Pot Odds":
            st.subheader("🧮 Understanding Pot Odds")
            
            st.write("""
            **Pot odds** help you decide if calling a bet is profitable:
            
            🔢 **Formula:** Pot Odds = Bet to Call ÷ (Current Pot + Bet to Call)
            
            📏 **Rule:** Call when your win chance is higher than the pot odds percentage
            
            **Example:** If there's $100 in the pot and you need to call $25:
            - Pot odds = $25 ÷ ($100 + $25) = 20%
            - You need at least 20% chance of winning to call profitably
            """)
            
            st.write("**🧮 Pot Odds Calculator:**")
            current_pot = st.number_input("Current pot size ($)", min_value=1, value=100)
            bet_to_call = st.number_input("Bet you need to call ($)", min_value=1, value=25)
            
            pot_odds = bet_to_call / (current_pot + bet_to_call)
            st.metric("Pot Odds", f"{pot_odds:.1%}")
            st.write(f"💡 You need at least **{pot_odds:.1%}** chance of winning to call profitably")
        
        else:
            st.info(f"📖 {lesson_choice} lesson coming soon! Check back later.")

    # Tab 4: Practice Mode
    with tab4:
        st.header("🎮 Practice Your Poker Skills")
        
        st.write("Test your poker knowledge with random scenarios!")
        
        if st.button("🆕 New Practice Hand", type="primary"):
            # Generate a random practice scenario
            practice_hand = []
            used_cards = set()
            
            while len(practice_hand) < 5:
                suit = random.randint(1, 4)
                rank = random.randint(1, 13)
                card = (suit, rank)
                if card not in used_cards:
                    practice_hand.append(card)
                    used_cards.add(card)
            
            # Create betting scenario
            pot_size = random.randint(20, 200)
            bet_size = random.randint(5, pot_size // 2)
            
            st.session_state.practice_scenario = {
                'cards': practice_hand,
                'pot': pot_size,
                'bet': bet_size,
                'hand_type': check_poker_hand(practice_hand)
            }
        
        if 'practice_scenario' in st.session_state:
            scenario = st.session_state.practice_scenario
            
            card_display = " ".join([f"{RANK_NAMES[rank]}{SUIT_SYMBOLS[suit]}" 
                                   for suit, rank in scenario['cards']])
            st.write(f"**Your Hand:** {card_display}")
            st.write(f"**Pot:** ${scenario['pot']} | **Bet to Call:** ${scenario['bet']}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Question 1: What hand do you have?**")
                hand_guess = st.selectbox("Hand type:", list(HAND_TYPES.values()), key="practice_hand")
            
            with col2:
                st.write("**Question 2: What should you do?**")
                action_guess = st.selectbox("Your action:", ["Fold", "Call", "Raise"], key="practice_action")
            
            if st.button("📝 Submit Answers"):
                # Check hand recognition
                correct_hand = HAND_TYPES[scenario['hand_type']]
                hand_correct = hand_guess == correct_hand
                
                # Check action decision
                hand_strength = HAND_STRENGTHS[scenario['hand_type']]
                pot_odds = scenario['bet'] / (scenario['pot'] + scenario['bet'])
                optimal_action, _ = decide_action(hand_strength, pot_odds)
                action_names = ["Fold", "Call", "Raise"]
                correct_action = action_names[optimal_action]
                action_correct = action_guess == correct_action
                
                # Update scores
                if hand_correct:
                    st.session_state.correct_guesses += 1
                if action_correct:
                    st.session_state.correct_guesses += 1
                
                st.session_state.hands_played += 1
                
                # Show results
                col1, col2 = st.columns(2)
                
                with col1:
                    if hand_correct:
                        st.success(f"✅ Hand: Correct! It's {correct_hand}")
                    else:
                        st.error(f"❌ Hand: Wrong. It's {correct_hand}")
                
                with col2:
                    if action_correct:
                        st.success(f"✅ Action: Correct! {correct_action} is optimal")
                    else:
                        st.error(f"❌ Action: Should {correct_action}")
                
                st.write("**📖 Explanations:**")
                hand_explanation = explain_hand(scenario['hand_type'], scenario['cards'])
                st.info(hand_explanation)
                
                action_explanation = explain_action(optimal_action, hand_strength, pot_odds)
                st.info(action_explanation)

else:
    # Welcome screen
    st.header("👋 Welcome to Poker AI!")
    
    st.write("""
    This app will help you learn poker and predict hand strengths using AI.
    
    **To get started:**
    1. 👈 Use the sidebar to either upload your own poker data or click "Use Demo Data"
    2. 🎯 Try the Hand Analyzer to see how strong different poker hands are
    3. 📚 Learn poker basics in the tutorial section
    4. 🎮 Practice your skills with random hands
    """)
    
    st.info("💡 **Tip:** Start with 'Use Demo Data' if you don't have your own poker data file!")









