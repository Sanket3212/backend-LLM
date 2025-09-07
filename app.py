from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import json
import os
import logging
import gc
import threading
import re
import requests
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, origins=['*'])

# Configure Gemini AI
try:
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        logger.error("GEMINI_API_KEY not found!")
        raise ValueError("GEMINI_API_KEY environment variable not set")
    
    genai.configure(api_key=api_key)
    
    # Configure generation settings for more consistent responses
    generation_config = {
        "temperature": 0.1,  # Lower temperature for more consistent responses
        "top_p": 0.9,
        "top_k": 40,
        "max_output_tokens": 1024,
    }
    
    model = genai.GenerativeModel(
        'gemini-1.5-flash',
        generation_config=generation_config
    )
    
    # Test the model
    test_response = model.generate_content("Hello")
    logger.info(f"Successfully configured Gemini AI")
        
except Exception as e:
    logger.error(f"Failed to configure Gemini AI: {e}")
    model = None

# Enhanced Menu with aliases and descriptions
MENU = {
    "Chicken Sandwich": {
        "price": 5.0,
        "aliases": ["chicken", "chicken sandwich", "chick sand", "chicken burger"],
        "description": "Grilled chicken sandwich with lettuce and mayo"
    },
    "Fries": {
        "price": 2.0,
        "aliases": ["fries", "french fries", "potato fries", "chips"],
        "description": "Golden crispy french fries"
    },
    "Coke": {
        "price": 1.5,
        "aliases": ["coke", "coca cola", "cola", "soda", "soft drink"],
        "description": "Refreshing Coca-Cola"
    },
    "Veggie Burger": {
        "price": 4.0,
        "aliases": ["veggie burger", "vegetarian burger", "veg burger", "plant burger"],
        "description": "Delicious plant-based burger"
    },
    "Pizza Slice": {
        "price": 3.0,
        "aliases": ["pizza", "pizza slice", "slice", "pizza piece"],
        "description": "Fresh cheese pizza slice"
    }
}

@dataclass
class OrderItem:
    name: str
    quantity: int
    unit_price: float
    total_price: float

@dataclass
class OrderTicket:
    ticket_id: str
    timestamp: datetime
    items: List[OrderItem]
    subtotal: float
    tax: float
    total: float
    status: str = "pending"
    mantis_ticket_id: Optional[int] = None
    mantis_url: Optional[str] = None

# MantisBT Integration Class
class MantisBTIntegration:
    """MantisBT integration for creating order tickets"""
    
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.headers = {
            'Authorization': api_key,
            'Content-Type': 'application/json'
        }
        self.api_endpoint = f"{self.base_url}/api/rest"
        
    def test_connection(self) -> bool:
        """Test connection to MantisBT API"""
        try:
            response = requests.get(
                f"{self.api_endpoint}/users/me",
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info("MantisBT connection successful")
                return True
            else:
                logger.error(f"MantisBT connection failed: {response.status_code} - {response.text}")
                return False
                
        except requests.RequestException as e:
            logger.error(f"MantisBT connection error: {e}")
            return False
    
    def get_projects(self) -> Optional[list]:
        """Get available projects"""
        try:
            response = requests.get(
                f"{self.api_endpoint}/projects",
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                projects = response.json()
                logger.info(f"Found {len(projects.get('projects', []))} projects")
                return projects.get('projects', [])
            else:
                logger.error(f"Failed to get projects: {response.status_code}")
                return None
                
        except requests.RequestException as e:
            logger.error(f"Error getting projects: {e}")
            return None
    
    def create_order_ticket(self, order_ticket: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create a MantisBT ticket for a confirmed order"""
        try:
            # Format order details
            summary = f"Food Order #{order_ticket['id']} - Total: ${order_ticket['total']:.2f}"
            
            # Create detailed description
            description = self._format_order_description(order_ticket)
            
            # Create Mantis ticket payload
            mantis_ticket = {
                "summary": summary,
                "description": description,
                "category": {"name": "Order Processing"},
                "priority": {"name": "normal"},
                "severity": {"name": "minor"},
                "project": {"id": 1},  # Default project ID
                "status": {"name": "new"},
                "tags": [
                    {"name": "food-order"},
                    {"name": f"order-{order_ticket['id']}"},
                    {"name": "automated"}
                ]
            }
            
            # Send request to MantisBT
            response = requests.post(
                f"{self.api_endpoint}/issues",
                headers=self.headers,
                json=mantis_ticket,
                timeout=15
            )
            
            if response.status_code == 201:
                mantis_response = response.json()
                logger.info(f"MantisBT ticket created successfully: {mantis_response.get('issue', {}).get('id')}")
                return {
                    "success": True,
                    "mantis_ticket_id": mantis_response.get('issue', {}).get('id'),
                    "mantis_url": f"{self.base_url}/view.php?id={mantis_response.get('issue', {}).get('id')}",
                    "summary": summary
                }
            else:
                logger.error(f"Failed to create MantisBT ticket: {response.status_code} - {response.text}")
                return {
                    "success": False,
                    "error": f"MantisBT API error: {response.status_code}",
                    "details": response.text
                }
                
        except requests.RequestException as e:
            logger.error(f"Error creating MantisBT ticket: {e}")
            return {
                "success": False,
                "error": f"Connection error: {str(e)}"
            }
        except Exception as e:
            logger.error(f"Unexpected error creating MantisBT ticket: {e}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}"
            }
    
    def _format_order_description(self, order_ticket: Dict[str, Any]) -> str:
        """Format order details for MantisBT description"""
        description = f"""
**Food Order Details**

**Order ID:** {order_ticket['id']}
**Timestamp:** {order_ticket['timestamp']}
**Status:** {order_ticket.get('status', 'Confirmed')}

**Items Ordered:**
"""
        
        for item in order_ticket.get('items', []):
            description += f"- {item['name']} x{item['qty']} @ ${item['unit_price']:.2f} each = ${item['price']:.2f}\n"
        
        description += f"""
**Order Summary:**
- Subtotal: ${order_ticket.get('subtotal', 0):.2f}
- Tax: ${order_ticket.get('tax', 0):.2f}
- **Total: ${order_ticket.get('total', 0):.2f}**

**Order Processing Notes:**
- Order was placed through the automated food ordering system
- Customer confirmation received
- Payment processing required
- Kitchen notification needed

**Next Actions:**
1. Process payment
2. Send order to kitchen
3. Update customer on order status
4. Prepare order for pickup/delivery

*This ticket was automatically created by the food ordering system.*
"""
        
        return description.strip()
    
    def update_ticket_status(self, ticket_id: int, status: str, note: str = "") -> bool:
        """Update MantisBT ticket status"""
        try:
            update_payload = {
                "status": {"name": status}
            }
            
            if note:
                update_payload["notes"] = [{"text": note}]
            
            response = requests.patch(
                f"{self.api_endpoint}/issues/{ticket_id}",
                headers=self.headers,
                json=update_payload,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"MantisBT ticket {ticket_id} updated to status: {status}")
                return True
            else:
                logger.error(f"Failed to update MantisBT ticket: {response.status_code}")
                return False
                
        except requests.RequestException as e:
            logger.error(f"Error updating MantisBT ticket: {e}")
            return False

# Initialize MantisBT integration
MANTIS_CONFIG = {
    "base_url": "https://sanketkumbhar.mantishub.io",
    "api_key": "hQNZgJjCafmXDAXCBuIrBz0cQInVvTYz"
}

mantis_integration = MantisBTIntegration(
    base_url=MANTIS_CONFIG["base_url"],
    api_key=MANTIS_CONFIG["api_key"]
)

class EnhancedOrderProcessor:
    def __init__(self):
        self.current_cart = []
        self.conversation_history = []
        self.ticket_counter = 1000
        
    def normalize_item_name(self, input_name: str) -> Optional[str]:
        """Normalize item name using aliases"""
        input_lower = input_name.lower().strip()
        
        for menu_item, details in MENU.items():
            if input_lower in [alias.lower() for alias in details["aliases"]]:
                return menu_item
        return None
    
    def extract_quantity_from_text(self, text: str) -> Dict[str, int]:
        """Extract quantities from natural language"""
        # Common quantity patterns
        quantity_patterns = [
            r'(\d+)\s*x?\s*([a-zA-Z\s]+)',  # "2 chicken sandwiches"
            r'([a-zA-Z\s]+)\s*x\s*(\d+)',   # "chicken sandwich x 2"
            r'(one|two|three|four|five|six|seven|eight|nine|ten)\s+([a-zA-Z\s]+)',
            r'a\s+([a-zA-Z\s]+)',  # "a chicken sandwich"
        ]
        
        word_to_number = {
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
            'a': 1, 'an': 1
        }
        
        extracted_items = {}
        text_lower = text.lower()
        
        # Try each pattern
        for pattern in quantity_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                if len(match) == 2:
                    qty_str, item_str = match
                    
                    # Convert quantity to number
                    if qty_str.isdigit():
                        qty = int(qty_str)
                    elif qty_str in word_to_number:
                        qty = word_to_number[qty_str]
                    else:
                        qty = 1
                    
                    # Normalize item name
                    normalized_name = self.normalize_item_name(item_str.strip())
                    if normalized_name:
                        extracted_items[normalized_name] = qty
        
        return extracted_items
    
    def update_cart(self, items: List[Dict[str, Any]]) -> None:
        self.current_cart = items
        
    def get_cart_total(self) -> float:
        return sum(item.get('price', 0) for item in self.current_cart)
    
    def add_item_to_cart(self, name: str, qty: int) -> None:
        unit_price = MENU.get(name, {}).get('price', 0)
        total_price = unit_price * qty
        
        # Check if item already exists
        for item in self.current_cart:
            if item['name'] == name:
                item['qty'] += qty
                item['price'] = unit_price * item['qty']
                return
                
        # Add new item
        self.current_cart.append({
            "name": name,
            "qty": qty,
            "unit_price": unit_price,
            "price": total_price
        })
    
    def remove_item_from_cart(self, name: str) -> None:
        self.current_cart = [item for item in self.current_cart if item['name'] != name]
    
    def update_item_quantity(self, name: str, qty: int) -> None:
        if qty <= 0:
            self.remove_item_from_cart(name)
            return
            
        unit_price = MENU.get(name, {}).get('price', 0)
        for item in self.current_cart:
            if item['name'] == name:
                item['qty'] = qty
                item['price'] = unit_price * qty
                return
    
    def generate_ticket(self) -> OrderTicket:
        """Generate Mantis-style order ticket"""
        if not self.current_cart:
            return None
        
        ticket_id = f"ORD-{self.ticket_counter}"
        self.ticket_counter += 1
        
        order_items = []
        subtotal = 0
        
        for cart_item in self.current_cart:
            item = OrderItem(
                name=cart_item['name'],
                quantity=cart_item['qty'],
                unit_price=cart_item['unit_price'],
                total_price=cart_item['price']
            )
            order_items.append(item)
            subtotal += cart_item['price']
        
        tax = round(subtotal * 0.08, 2)  # 8% tax
        total = round(subtotal + tax, 2)
        
        ticket = OrderTicket(
            ticket_id=ticket_id,
            timestamp=datetime.now(),
            items=order_items,
            subtotal=subtotal,
            tax=tax,
            total=total
        )
        
        return ticket

order_processor = EnhancedOrderProcessor()

def create_enhanced_prompt(user_message: str, conversation_context: List = None) -> str:
    """Create enhanced prompt with better context and examples"""
    
    menu_text = ""
    for item, details in MENU.items():
        aliases_str = ", ".join(details["aliases"][:3])  # Show first 3 aliases
        menu_text += f"- {item}: ${details['price']:.2f} (also: {aliases_str})\n"
    
    context_text = ""
    if conversation_context:
        recent_messages = conversation_context[-3:]  # Last 3 messages
        context_text = f"RECENT CONVERSATION:\n{json.dumps(recent_messages, indent=2)}\n\n"
    
    cart_text = f"CURRENT CART: {json.dumps(order_processor.current_cart, indent=2)}\n\n" if order_processor.current_cart else "CURRENT CART: Empty\n\n"
    
    prompt = f"""You are an expert food ordering assistant with advanced natural language understanding.

{context_text}MENU:
{menu_text}
{cart_text}USER MESSAGE: "{user_message}"

INSTRUCTIONS:
1. Analyze the user's intent carefully
2. Extract food items and quantities precisely
3. Handle variations in item names (use aliases)
4. Calculate prices accurately
5. Maintain conversation flow

INTENT TYPES:
- "order_food": New order (clears cart)
- "add_item": Add to existing cart
- "remove_item": Remove specific items
- "update_quantity": Change quantities
- "ask_menu": Show menu
- "finalize_order": Complete order
- "view_cart": Show current cart
- "modify_order": General modifications

ENTITY EXTRACTION RULES:
- "2 chicken sandwiches" → name: "Chicken Sandwich", qty: 2
- "a coke" → name: "Coke", qty: 1
- "remove fries" → name: "Fries" (for removal)
- Handle synonyms: "burger" → "Veggie Burger", "soda" → "Coke"

RESPONSE FORMAT (JSON only):
{{
  "intent": "one_of_the_intents_above",
  "items": [
    {{
      "name": "Exact Menu Item Name",
      "qty": number,
      "unit_price": number,
      "price": total_price_for_quantity
    }}
  ],
  "total": calculated_total,
  "message": "Helpful confirmation message",
  "confidence": 0.95,
  "extracted_entities": ["entity1", "entity2"]
}}

EXAMPLES:
User: "I want 2 chicken sandwiches and fries"
Response: {{"intent": "order_food", "items": [{{"name": "Chicken Sandwich", "qty": 2, "unit_price": 5.0, "price": 10.0}}, {{"name": "Fries", "qty": 1, "unit_price": 2.0, "price": 2.0}}], "total": 12.0, "message": "Added 2 Chicken Sandwiches and 1 Fries to your order. Total: $12.00", "confidence": 0.98}}

User: "Add a coke"
Response: {{"intent": "add_item", "items": [{{"name": "Coke", "qty": 1, "unit_price": 1.5, "price": 1.5}}], "total": 13.5, "message": "Added 1 Coke to your cart. New total: $13.50", "confidence": 0.95}}

IMPORTANT: Return ONLY valid JSON. No explanations outside JSON."""
    
    return prompt

# API Routes

@app.route('/health', methods=['GET'])
def health_check():
    """Enhanced health check with MantisBT status"""
    mantis_status = mantis_integration.test_connection()
    
    return jsonify({
        "status": "healthy", 
        "service": "enhanced-food-ordering-api",
        "gemini_configured": model is not None,
        "mantis_configured": mantis_status,
        "menu_items": len(MENU),
        "cart_items": len(order_processor.current_cart),
        "features": [
            "enhanced_nlp", 
            "mantis_ticketing", 
            "conversation_context",
            "automated_tracking"
        ],
        "integrations": {
            "gemini_ai": model is not None,
            "mantis_bt": mantis_status
        }
    })

@app.route('/menu', methods=['GET'])
def get_menu():
    """Get enhanced restaurant menu with descriptions"""
    menu_items = []
    for name, details in MENU.items():
        menu_items.append({
            "name": name, 
            "price": details["price"],
            "description": details["description"],
            "aliases": details["aliases"][:3]  # First 3 aliases
        })
    return jsonify({"menu": menu_items})

@app.route('/process-order', methods=['POST'])
def process_order():
    """Process natural language order with enhanced LLM"""
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
        
        # Add to conversation history
        order_processor.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "message": user_message,
            "type": "user"
        })
        
        # Create enhanced prompt
        enhanced_prompt = create_enhanced_prompt(
            user_message, 
            order_processor.conversation_history
        )
        
        # Generate response with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = model.generate_content(enhanced_prompt)
                response_text = response.text.strip()
                
                logger.info(f"AI Response (attempt {attempt + 1}): {response_text}")
                
                # Clean and extract JSON more robustly
                response_text = re.sub(r'```(?:json)?', '', response_text).strip()
                
                # Find JSON boundaries more accurately
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if not json_match:
                    raise ValueError("No JSON found in response")
                
                json_str = json_match.group()
                parsed_response = json.loads(json_str)
                
                # Validate response structure
                required_fields = ['intent', 'items', 'total', 'message']
                if not all(field in parsed_response for field in required_fields):
                    raise ValueError(f"Missing required fields. Got: {list(parsed_response.keys())}")
                
                break  # Success, exit retry loop
                
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    # Fallback: Try simple entity extraction
                    return fallback_processing(user_message)
                continue
        
        # Process the validated response
        intent = parsed_response.get('intent', '')
        items = parsed_response.get('items', [])
        
        # Handle different intents
        if intent == 'ask_menu':
            response_data = {
                "intent": intent,
                "items": [],
                "total": 0,
                "menu": [{"name": name, "price": details["price"], "description": details["description"]} 
                        for name, details in MENU.items()],
                "message": "Here's our menu! What would you like to order?",
                "cart": order_processor.current_cart
            }
        
        elif intent == 'view_cart':
            actual_total = order_processor.get_cart_total()
            response_data = {
                "intent": intent,
                "items": order_processor.current_cart,
                "total": actual_total,
                "message": f"Your cart has {len(order_processor.current_cart)} items. Total: ${actual_total:.2f}",
                "cart": order_processor.current_cart
            }
        
        elif intent == 'finalize_order':
            ticket = order_processor.generate_ticket()
            if ticket:
                # Prepare ticket data for MantisBT
                ticket_data = {
                    "id": ticket.ticket_id,
                    "timestamp": ticket.timestamp.isoformat(),
                    "items": [
                        {
                            "name": item.name,
                            "qty": item.quantity,
                            "unit_price": item.unit_price,
                            "price": item.total_price
                        }
                        for item in ticket.items
                    ],
                    "subtotal": ticket.subtotal,
                    "tax": ticket.tax,
                    "total": ticket.total,
                    "status": "confirmed"
                }
                
                # Create MantisBT ticket
                mantis_result = None
                if mantis_integration.test_connection():
                    mantis_result = mantis_integration.create_order_ticket(ticket_data)
                    if mantis_result and mantis_result.get("success"):
                        ticket.mantis_ticket_id = mantis_result.get("mantis_ticket_id")
                        ticket.mantis_url = mantis_result.get("mantis_url")
                        logger.info(f"Created MantisBT ticket #{ticket.mantis_ticket_id} for order {ticket.ticket_id}")
                
                response_data = {
                    "intent": intent,
                    "items": order_processor.current_cart,
                    "total": ticket.total,
                    "message": f"Order finalized! Ticket #{ticket.ticket_id}",
                    "ticket": {
                        "id": ticket.ticket_id,
                        "timestamp": ticket.timestamp.isoformat(),
                        "subtotal": ticket.subtotal,
                        "tax": ticket.tax,
                        "total": ticket.total,
                        "items": [{"name": item.name, "qty": item.quantity, "price": item.total_price} 
                                for item in ticket.items]
                    }
                }
                
                # Add MantisBT information if successful
                if mantis_result and mantis_result.get("success"):
                    response_data["mantis"] = {
                        "ticket_id": mantis_result.get("mantis_ticket_id"),
                        "url": mantis_result.get("mantis_url"),
                        "status": "created",
                        "message": f"MantisBT ticket #{mantis_result.get('mantis_ticket_id')} created"
                    }
                    response_data["message"] += f" MantisBT tracking ticket #{mantis_result.get('mantis_ticket_id')} created."
                elif mantis_result:
                    response_data["mantis"] = {
                        "status": "failed",
                        "error": mantis_result.get("error", "Unknown error"),
                        "message": "Failed to create MantisBT ticket"
                    }
                    response_data["message"] += " (Note: MantisBT ticket creation failed)"
                else:
                    response_data["mantis"] = {
                        "status": "unavailable",
                        "message": "MantisBT service unavailable"
                    }
                
                # Clear cart after finalizing
                order_processor.current_cart = []
            else:
                response_data = {
                    "intent": intent,
                    "items": [],
                    "total": 0,
                    "message": "Your cart is empty. Please add items before finalizing.",
                    "cart": []
                }
        
        else:
            # Handle cart modifications
            if intent == 'order_food':
                order_processor.current_cart = []
                
            for item in items:
                item_name = item.get('name', '')
                qty = item.get('qty', 1)
                
                # Normalize item name
                normalized_name = order_processor.normalize_item_name(item_name)
                if not normalized_name:
                    continue  # Skip invalid items
                
                if intent in ['order_food', 'add_item']:
                    order_processor.add_item_to_cart(normalized_name, qty)
                elif intent == 'remove_item':
                    order_processor.remove_item_from_cart(normalized_name)
                elif intent in ['update_quantity', 'modify_order']:
                    order_processor.update_item_quantity(normalized_name, qty)
            
            actual_total = order_processor.get_cart_total()
            response_data = {
                "intent": intent,
                "items": order_processor.current_cart,
                "total": actual_total,
                "message": parsed_response.get('message', f"Order processed! Total: ${actual_total:.2f}"),
                "confidence": parsed_response.get('confidence', 0.9),
                "cart": order_processor.current_cart
            }
        
        # Add to conversation history
        order_processor.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "response": response_data,
            "type": "assistant"
        })
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Processing error: {e}")
        return jsonify({"error": f"Processing error: {str(e)}"}), 500

def fallback_processing(user_message: str):
    """Fallback processing when LLM fails"""
    logger.info("Using fallback processing")
    
    # Simple keyword-based extraction
    extracted_items = order_processor.extract_quantity_from_text(user_message)
    
    if extracted_items:
        for item_name, qty in extracted_items.items():
            order_processor.add_item_to_cart(item_name, qty)
        
        total = order_processor.get_cart_total()
        return jsonify({
            "intent": "add_item",
            "items": order_processor.current_cart,
            "total": total,
            "message": f"Added items to cart (fallback mode). Total: ${total:.2f}",
            "fallback": True
        })
    else:
        return jsonify({
            "intent": "ask_menu",
            "items": [],
            "total": 0,
            "message": "I didn't understand that. Here's our menu:",
            "menu": [{"name": name, "price": details["price"]} for name, details in MENU.items()],
            "fallback": True
        })

@app.route('/cart', methods=['GET'])
def get_cart():
    """Get current cart with enhanced details"""
    total = order_processor.get_cart_total()
    tax = round(total * 0.08, 2)
    final_total = round(total + tax, 2)
    
    return jsonify({
        "items": order_processor.current_cart,
        "subtotal": total,
        "tax": tax,
        "total": final_total,
        "item_count": len(order_processor.current_cart)
    })

@app.route('/cart/clear', methods=['POST'])
def clear_cart():
    """Clear the current cart"""
    order_processor.current_cart = []
    order_processor.conversation_history = []
    return jsonify({
        "message": "Cart cleared successfully",
        "items": [],
        "total": 0
    })

@app.route('/finalize', methods=['POST'])
def finalize_order():
    """Generate final order ticket and create MantisBT ticket"""
    ticket = order_processor.generate_ticket()
    
    if not ticket:
        return jsonify({"error": "Cart is empty"}), 400
    
    # Prepare ticket data for MantisBT
    ticket_data = {
        "id": ticket.ticket_id,
        "timestamp": ticket.timestamp.isoformat(),
        "items": [
            {
                "name": item.name,
                "qty": item.quantity,
                "unit_price": item.unit_price,
                "price": item.total_price
            }
            for item in ticket.items
        ],
        "subtotal": ticket.subtotal,
        "tax": ticket.tax,
        "total": ticket.total,
        "status": "confirmed"
    }
    
    # Create MantisBT ticket
    mantis_result = None
    if mantis_integration.test_connection():
        mantis_result = mantis_integration.create_order_ticket(ticket_data)
        if mantis_result and mantis_result.get("success"):
            ticket.mantis_ticket_id = mantis_result.get("mantis_ticket_id")
            ticket.mantis_url = mantis_result.get("mantis_url")
            logger.info(f"Created MantisBT ticket #{ticket.mantis_ticket_id} for order {ticket.ticket_id}")
    
    # Clear cart after finalizing
    order_processor.current_cart = []
    
    # Prepare response
    response_data = {
        "ticket": {
            "id": ticket.ticket_id,
            "timestamp": ticket.timestamp.isoformat(),
            "items": [
                {
                    "name": item.name,
                    "quantity": item.quantity,
                    "unit_price": item.unit_price,
                    "total_price": item.total_price
                }
                for item in ticket.items
            ],
            "subtotal": ticket.subtotal,
            "tax": ticket.tax,
            "total": ticket.total,
            "status": ticket.status
        },
        "message": f"Order ticket #{ticket.ticket_id} generated successfully!"
    }
    
    # Add MantisBT information if successful
    if mantis_result and mantis_result.get("success"):
        response_data["mantis"] = {
            "ticket_id": mantis_result.get("mantis_ticket_id"),
            "url": mantis_result.get("mantis_url"),
            "status": "created",
            "message": f"MantisBT ticket #{mantis_result.get('mantis_ticket_id')} created"
        }
        response_data["message"] += f" MantisBT tracking ticket #{mantis_result.get('mantis_ticket_id')} created."
    elif mantis_result:
        response_data["mantis"] = {
            "status": "failed",
            "error": mantis_result.get("error", "Unknown error"),
            "message": "Failed to create MantisBT ticket"
        }
        response_data["message"] += " (Note: MantisBT ticket creation failed)"
    else:
        response_data["mantis"] = {
            "status": "unavailable",
            "message": "MantisBT service unavailable"
        }
    
    return jsonify(response_data)

# MantisBT specific endpoints
@app.route('/mantis/status', methods=['GET'])
def mantis_status():
    """Check MantisBT connection status"""
    try:
        connection_ok = mantis_integration.test_connection()
        projects = mantis_integration.get_projects() if connection_ok else None
        
        return jsonify({
            "mantis_connected": connection_ok,
            "base_url": MANTIS_CONFIG["base_url"],
            "api_configured": bool(MANTIS_CONFIG["api_key"]),
            "status": "connected" if connection_ok else "disconnected",
            "projects_count": len(projects) if projects else 0,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            "mantis_connected": False,
            "error": str(e),
            "status": "error",
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/mantis/test', methods=['POST'])
def test_mantis_ticket():
    """Test MantisBT ticket creation"""
    try:
        # Create a test order
        test_order = {
            "id": f"TEST-{int(datetime.now().timestamp())}",
            "timestamp": datetime.now().isoformat(),
            "items": [
                {"name": "Test Chicken Sandwich", "qty": 1, "unit_price": 5.0, "price": 5.0},
                {"name": "Test Fries", "qty": 1, "unit_price": 2.0, "price": 2.0}
            ],
            "subtotal": 7.0,
            "tax": 0.56,
            "total": 7.56,
            "status": "test"
        }
        
        result = mantis_integration.create_order_ticket(test_order)
        
        if result and result.get("success"):
            return jsonify({
                "success": True,
                "mantis_ticket_id": result.get("mantis_ticket_id"),
                "mantis_url": result.get("mantis_url"),
                "message": "Test ticket created successfully",
                "test_order": test_order
            })
        else:
            return jsonify({
                "success": False,
                "error": result.get("error", "Unknown error") if result else "No result",
                "message": "Failed to create test ticket"
            }), 500
            
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "message": "Error creating test ticket"
        }), 500

@app.route('/mantis/projects', methods=['GET'])
def get_mantis_projects():
    """Get MantisBT projects"""
    try:
        if not mantis_integration.test_connection():
            return jsonify({"error": "Cannot connect to MantisBT"}), 503
            
        projects = mantis_integration.get_projects()
        
        if projects:
            return jsonify({
                "success": True,
                "projects": projects,
                "count": len(projects)
            })
        else:
            return jsonify({
                "success": False,
                "error": "Failed to retrieve projects"
            }), 500
            
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/mantis/update/<int:ticket_id>', methods=['POST'])
def update_mantis_ticket(ticket_id):
    """Update MantisBT ticket status"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        status = data.get('status', 'acknowledged')
        note = data.get('note', '')
        
        success = mantis_integration.update_ticket_status(ticket_id, status, note)
        
        if success:
            return jsonify({
                "success": True,
                "message": f"Ticket {ticket_id} updated to status: {status}"
            })
        else:
            return jsonify({
                "success": False,
                "error": "Failed to update ticket"
            }), 500
            
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# Enhanced statistics endpoint
@app.route('/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    return jsonify({
        "system": {
            "status": "running",
            "uptime": "N/A",  # You can implement uptime tracking
            "timestamp": datetime.now().isoformat()
        },
        "menu": {
            "total_items": len(MENU),
            "items": list(MENU.keys())
        },
        "cart": {
            "items_count": len(order_processor.current_cart),
            "total_value": order_processor.get_cart_total(),
            "items": order_processor.current_cart
        },
        "conversation": {
            "messages_count": len(order_processor.conversation_history)
        },
        "integrations": {
            "gemini_ai": model is not None,
            "mantis_bt": mantis_integration.test_connection()
        },
        "orders": {
            "last_ticket_number": order_processor.ticket_counter - 1
        }
    })

# Error handlers
@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({
        "error": "Internal server error",
        "timestamp": datetime.now().isoformat()
    }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found",
        "available_endpoints": [
            "/health", "/menu", "/process-order", "/cart", "/cart/clear", 
            "/finalize", "/mantis/status", "/mantis/test", "/mantis/projects", 
            "/mantis/update/<ticket_id>", "/stats"
        ],
        "timestamp": datetime.now().isoformat()
    }), 404

@app.errorhandler(400)
def bad_request(error):
    return jsonify({
        "error": "Bad request",
        "message": "Please check your request format",
        "timestamp": datetime.now().isoformat()
    }), 400

# Utility function to create mantis ticket for order (can be used programmatically)
def create_mantis_ticket_for_order(order_ticket: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create MantisBT ticket for confirmed order
    
    Args:
        order_ticket: Order ticket data from the food ordering system
        
    Returns:
        Dict containing success status and MantisBT ticket details
    """
    
    # Test connection first
    if not mantis_integration.test_connection():
        return {
            "success": False,
            "error": "Unable to connect to MantisBT",
            "mantis_ticket_id": None
        }
    
    # Create the ticket
    result = mantis_integration.create_order_ticket(order_ticket)
    
    if result and result.get("success"):
        logger.info(f"Successfully created MantisBT ticket for order {order_ticket['id']}")
        return {
            "success": True,
            "mantis_ticket_id": result.get("mantis_ticket_id"),
            "mantis_url": result.get("mantis_url"),
            "summary": result.get("summary"),
            "message": f"Order ticket created in MantisBT: #{result.get('mantis_ticket_id')}"
        }
    else:
        logger.error(f"Failed to create MantisBT ticket: {result}")
        return {
            "success": False,
            "error": result.get("error", "Unknown error") if result else "No result",
            "mantis_ticket_id": None,
            "message": "Failed to create MantisBT ticket"
        }

# Main application startup
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV', 'production') == 'development'
    
    logger.info(f"Starting enhanced food ordering server on port {port}")
    logger.info(f"MantisBT URL: {MANTIS_CONFIG['base_url']}")
    logger.info(f"Gemini AI configured: {model is not None}")
    
    # Test MantisBT connection on startup
    if mantis_integration.test_connection():
        logger.info("✅ MantisBT connection successful on startup")
    else:
        logger.warning("⚠️ MantisBT connection failed on startup")
    
    app.run(host='0.0.0.0', port=port, debug=debug_mode)