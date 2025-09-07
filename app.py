from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import json
import os
from typing import Dict, List, Any

app = Flask(__name__)
CORS(app)

# Configure Gemini AI
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
model = genai.GenerativeModel('gemini-pro')

# Menu data
MENU = {
    "Chicken Sandwich": 5.0,
    "Fries": 2.0,
    "Coke": 1.5,
    "Veggie Burger": 4.0,
    "Pizza Slice": 3.0
}

# System prompt for the LLM
SYSTEM_PROMPT = """You are a food ordering assistant for a restaurant.
Menu:
- Chicken Sandwich - $5
- Fries - $2
- Coke - $1.5
- Veggie Burger - $4
- Pizza Slice - $3

Your job:
- Understand the user's natural language messages.
- Determine the intent: one of [order_food, add_item, remove_item, update_quantity, ask_menu, finalize_order].
- Extract ordered items with their name, quantity, and price.
- Always calculate and include the total price.
- When intent is ask_menu, return an empty item list and total = 0.
- When intent is finalize_order, return the current cart items with total.
- Only output valid JSON in the format below. Never add explanations or text outside JSON.

Output format:
{
  "intent": "order_food | add_item | remove_item | update_quantity | ask_menu | finalize_order",
  "items": [
    {"name": "Chicken Sandwich", "qty": 2, "price": 10},
    {"name": "Coke", "qty": 1, "price": 1.5}
  ],
  "total": 11.5
}

Current user message: """

class OrderProcessor:
    def __init__(self):
        self.current_cart = []
        
    def update_cart(self, items: List[Dict[str, Any]]) -> None:
        """Update the current cart with new items"""
        self.current_cart = items
        
    def get_cart_total(self) -> float:
        """Calculate total price of items in cart"""
        return sum(item.get('price', 0) for item in self.current_cart)
    
    def add_item_to_cart(self, name: str, qty: int) -> None:
        """Add item to cart or update quantity if exists"""
        unit_price = MENU.get(name, 0)
        total_price = unit_price * qty
        
        # Check if item already exists in cart
        for item in self.current_cart:
            if item['name'] == name:
                item['qty'] += qty
                item['price'] = unit_price * item['qty']
                return
                
        # Add new item to cart
        self.current_cart.append({
            "name": name,
            "qty": qty,
            "price": total_price
        })
    
    def remove_item_from_cart(self, name: str) -> None:
        """Remove item from cart"""
        self.current_cart = [item for item in self.current_cart if item['name'] != name]
    
    def update_item_quantity(self, name: str, qty: int) -> None:
        """Update quantity of item in cart"""
        if qty <= 0:
            self.remove_item_from_cart(name)
            return
            
        unit_price = MENU.get(name, 0)
        for item in self.current_cart:
            if item['name'] == name:
                item['qty'] = qty
                item['price'] = unit_price * qty
                return

# Global order processor instance
order_processor = OrderProcessor()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "food-ordering-api"})

@app.route('/menu', methods=['GET'])
def get_menu():
    """Get restaurant menu"""
    menu_items = []
    for name, price in MENU.items():
        menu_items.append({"name": name, "price": price})
    return jsonify({"menu": menu_items})

@app.route('/process-order', methods=['POST'])
def process_order():
    """Process natural language order using Gemini LLM"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({"error": "Message is required"}), 400
        
        # Create prompt with current cart context
        cart_context = f"\nCurrent cart: {json.dumps(order_processor.current_cart)}\n"
        full_prompt = SYSTEM_PROMPT + cart_context + user_message
        
        # Generate response using Gemini
        response = model.generate_content(full_prompt)
        
        # Extract JSON from response
        response_text = response.text.strip()
        
        # Try to find JSON in the response
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        
        if json_start == -1 or json_end == 0:
            return jsonify({"error": "Invalid response format from LLM"}), 500
            
        json_str = response_text[json_start:json_end]
        parsed_response = json.loads(json_str)
        
        # Validate response structure
        if not all(key in parsed_response for key in ['intent', 'items', 'total']):
            return jsonify({"error": "Invalid response structure"}), 500
        
        intent = parsed_response['intent']
        items = parsed_response['items']
        
        # Process based on intent
        if intent == 'ask_menu':
            menu_items = []
            for name, price in MENU.items():
                menu_items.append({"name": name, "price": price})
            return jsonify({
                "intent": intent,
                "items": [],
                "total": 0,
                "menu": menu_items,
                "message": "Here's our menu!"
            })
            
        elif intent == 'order_food':
            # Replace cart with new items
            order_processor.current_cart = []
            for item in items:
                if item['name'] in MENU:
                    order_processor.add_item_to_cart(item['name'], item['qty'])
                    
        elif intent == 'add_item':
            # Add items to existing cart
            for item in items:
                if item['name'] in MENU:
                    order_processor.add_item_to_cart(item['name'], item['qty'])
                    
        elif intent == 'remove_item':
            # Remove items from cart
            for item in items:
                order_processor.remove_item_from_cart(item['name'])
                
        elif intent == 'update_quantity':
            # Update quantities
            for item in items:
                if item['name'] in MENU:
                    order_processor.update_item_quantity(item['name'], item['qty'])
                    
        elif intent == 'finalize_order':
            # Return current cart for finalization
            pass
        
        # Calculate actual total
        actual_total = order_processor.get_cart_total()
        
        return jsonify({
            "intent": intent,
            "items": order_processor.current_cart,
            "total": actual_total,
            "message": f"Order processed successfully! Intent: {intent}"
        })
        
    except json.JSONDecodeError as e:
        return jsonify({"error": f"JSON parsing error: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"Processing error: {str(e)}"}), 500

@app.route('/cart', methods=['GET'])
def get_cart():
    """Get current cart contents"""
    return jsonify({
        "items": order_processor.current_cart,
        "total": order_processor.get_cart_total()
    })

@app.route('/cart/clear', methods=['POST'])
def clear_cart():
    """Clear the current cart"""
    order_processor.current_cart = []
    return jsonify({
        "message": "Cart cleared successfully",
        "items": [],
        "total": 0
    })

if __name__ == '__main__':
    # Check for required environment variable
    if not os.getenv('GEMINI_API_KEY'):
        print("Warning: GEMINI_API_KEY environment variable not set")
    
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)