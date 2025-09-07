# Updated app.py - Fix for Gemini model error

from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import json
import os
import logging
import gc
import threading
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, origins=['*'])  # Allow all origins for now

# Configure Gemini AI with updated model names
try:
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        logger.error("GEMINI_API_KEY not found!")
        raise ValueError("GEMINI_API_KEY environment variable not set")
    
    genai.configure(api_key=api_key)
    
    # Updated model names - try these in order of preference
    model_names = [
        'gemini-1.5-flash',      # Latest and fastest
        'gemini-1.5-pro',        # More capable but slower
        'gemini-pro',            # Legacy fallback
        'models/gemini-pro'      # Full path fallback
    ]
    
    model = None
    for model_name in model_names:
        try:
            model = genai.GenerativeModel(model_name)
            # Test the model with a simple request
            test_response = model.generate_content("Hello")
            logger.info(f"Successfully configured Gemini AI with model: {model_name}")
            break
        except Exception as e:
            logger.warning(f"Failed to configure model {model_name}: {e}")
            continue
    
    if model is None:
        raise Exception("Failed to configure any Gemini model")
        
except Exception as e:
    logger.error(f"Failed to configure Gemini AI: {e}")
    model = None

# Menu data
MENU = {
    "Chicken Sandwich": 5.0,
    "Fries": 2.0,
    "Coke": 1.5,
    "Veggie Burger": 4.0,
    "Pizza Slice": 3.0
}

# Global order processor
class OrderProcessor:
    def __init__(self):
        self.current_cart = []
        
    def update_cart(self, items: List[Dict[str, Any]]) -> None:
        self.current_cart = items
        
    def get_cart_total(self) -> float:
        return sum(item.get('price', 0) for item in self.current_cart)
    
    def add_item_to_cart(self, name: str, qty: int) -> None:
        unit_price = MENU.get(name, 0)
        total_price = unit_price * qty
        
        for item in self.current_cart:
            if item['name'] == name:
                item['qty'] += qty
                item['price'] = unit_price * item['qty']
                return
                
        self.current_cart.append({
            "name": name,
            "qty": qty,
            "price": total_price
        })
    
    def remove_item_from_cart(self, name: str) -> None:
        self.current_cart = [item for item in self.current_cart if item['name'] != name]
    
    def update_item_quantity(self, name: str, qty: int) -> None:
        if qty <= 0:
            self.remove_item_from_cart(name)
            return
            
        unit_price = MENU.get(name, 0)
        for item in self.current_cart:
            if item['name'] == name:
                item['qty'] = qty
                item['price'] = unit_price * qty
                return

order_processor = OrderProcessor()

# Memory cleanup
def cleanup_memory():
    while True:
        try:
            gc.collect()
            threading.Event().wait(300)  # Clean every 5 minutes
        except:
            break

cleanup_thread = threading.Thread(target=cleanup_memory, daemon=True)
cleanup_thread.start()

@app.route('/health', methods=['GET'])
def health_check():
    """Enhanced health check"""
    return jsonify({
        "status": "healthy", 
        "service": "food-ordering-api",
        "gemini_configured": model is not None,
        "menu_items": len(MENU),
        "cart_items": len(order_processor.current_cart)
    })

@app.route('/menu', methods=['GET'])
def get_menu():
    """Get restaurant menu"""
    menu_items = []
    for name, price in MENU.items():
        menu_items.append({"name": name, "price": price})
    return jsonify({"menu": menu_items})

@app.route('/test-ai', methods=['GET'])
def test_ai():
    """Test AI connectivity"""
    if not model:
        return jsonify({"error": "AI service not available"}), 503
    
    try:
        response = model.generate_content("Say hello")
        return jsonify({
            "status": "AI working", 
            "response": response.text,
            "model_working": True
        })
    except Exception as e:
        return jsonify({"error": f"AI test failed: {str(e)}"}), 500

@app.route('/process-order', methods=['POST'])
def process_order():
    """Process natural language order using Gemini LLM"""
    try:
        if not model:
            return jsonify({"error": "AI service not available. Please check GEMINI_API_KEY."}), 503
        
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({"error": "Message is required"}), 400
        
        logger.info(f"Processing message: {user_message}")
        
        # Enhanced system prompt with better formatting
        system_prompt = f"""You are a food ordering assistant for a restaurant. 

MENU:
- Chicken Sandwich: $5.00
- Fries: $2.00  
- Coke: $1.50
- Veggie Burger: $4.00
- Pizza Slice: $3.00

CURRENT CART: {json.dumps(order_processor.current_cart)}

USER MESSAGE: "{user_message}"

Analyze the user's message and determine their intent. Respond ONLY with valid JSON in this exact format:

{{
  "intent": "order_food|add_item|remove_item|update_quantity|ask_menu|finalize_order",
  "items": [
    {{"name": "Chicken Sandwich", "qty": 2, "price": 10.0}}
  ],
  "total": 10.0,
  "message": "Order processed successfully"
}}

RULES:
- Use exact menu item names
- Calculate correct prices (price = unit_price × quantity)  
- Only include valid menu items
- For "ask_menu" intent, return empty items array
- Be precise with quantities and calculations"""
        
        # Generate response using Gemini with enhanced error handling
        try:
            response = model.generate_content(system_prompt)
            response_text = response.text.strip()
            
            logger.info(f"AI Response: {response_text}")
            
            # Clean and extract JSON
            response_text = response_text.replace('```json', '').replace('```', '').strip()
            
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")
                
            json_str = response_text[json_start:json_end]
            parsed_response = json.loads(json_str)
            
            # Validate required fields
            if 'intent' not in parsed_response:
                raise ValueError("Missing 'intent' in AI response")
            
            # Process the order
            intent = parsed_response.get('intent', '')
            items = parsed_response.get('items', [])
            
            if intent == 'ask_menu':
                return jsonify({
                    "intent": intent,
                    "items": [],
                    "total": 0,
                    "menu": [{"name": name, "price": price} for name, price in MENU.items()],
                    "message": "Here's our menu! You can order by typing something like 'I want 2 chicken sandwiches'"
                })
            elif intent == 'order_food':
                order_processor.current_cart = []
                for item in items:
                    if item['name'] in MENU:
                        order_processor.add_item_to_cart(item['name'], item['qty'])
            elif intent == 'add_item':
                for item in items:
                    if item['name'] in MENU:
                        order_processor.add_item_to_cart(item['name'], item['qty'])
            elif intent == 'remove_item':
                for item in items:
                    order_processor.remove_item_from_cart(item['name'])
            elif intent == 'update_quantity':
                for item in items:
                    if item['name'] in MENU:
                        order_processor.update_item_quantity(item['name'], item['qty'])
            
            actual_total = order_processor.get_cart_total()
            
            return jsonify({
                "intent": intent,
                "items": order_processor.current_cart,
                "total": actual_total,
                "message": parsed_response.get('message', f"Order processed successfully! Intent: {intent}")
            })
            
        except json.JSONDecodeError as json_error:
            logger.error(f"JSON parsing error: {json_error}")
            logger.error(f"Raw response: {response_text}")
            return jsonify({"error": f"AI returned invalid JSON: {str(json_error)}"}), 500
            
        except Exception as ai_error:
            logger.error(f"AI processing error: {ai_error}")
            return jsonify({"error": f"AI processing failed: {str(ai_error)}"}), 500
            
    except Exception as e:
        logger.error(f"Processing error: {e}")
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

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV', 'production') == 'development'
    logger.info(f"Starting server on port {port}, debug={debug_mode}")
    app.run(host='0.0.0.0', port=port, debug=debug_mode)