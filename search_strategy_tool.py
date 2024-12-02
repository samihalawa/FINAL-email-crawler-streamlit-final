import json
from typing import List, Dict

class CustomerProfileSearchTool:
    """
    A tool for generating and managing search terms based on customer profiles and business pain points.
    Supports multiple languages and provides targeted search term generation for different business scenarios.
    """
    def __init__(self):
        self.business_pain_points = {
            "productivity": [
                "cómo reducir tiempo en tareas administrativas",
                "automatizar trabajo repetitivo empresa",
                "mejorar eficiencia procesos empresa",
                "reducir costes operativos negocio",
                "optimizar gestión tiempo empresa"
            ],
            "customer_service": [
                "mejorar atención cliente 24/7",
                "reducir tiempo respuesta clientes",
                "gestionar alto volumen consultas",
                "problemas respuesta rápida clientes",
                "optimizar servicio cliente sin contratar"
            ],
            "data_analysis": [
                "cómo analizar datos ventas",
                "mejorar predicción inventario",
                "analizar comportamiento cliente",
                "problemas análisis datos empresa",
                "optimizar decisiones negocio datos"
            ],
            "marketing": [
                "personalizar marketing clientes",
                "mejorar conversión ventas",
                "automatizar campañas email",
                "problemas segmentación clientes",
                "optimizar estrategia marketing"
            ]
        }
        
        self.professional_profiles = {
            "business_owner": [
                "dueño negocio pequeño",
                "emprendedor startup",
                "director empresa pyme",
                "fundador empresa nueva",
                "empresario independiente"
            ],
            "manager": [
                "director operaciones",
                "gerente ventas",
                "jefe proyecto",
                "responsable marketing",
                "coordinador equipo"
            ],
            "consultant": [
                "consultor empresarial",
                "asesor negocios",
                "coach ejecutivo",
                "mentor startups",
                "consultor estratégico"
            ]
        }

    def get_search_terms_by_pain_point(self, pain_point: str) -> List[str]:
        """Get search terms for a specific business pain point."""
        return self.business_pain_points.get(pain_point, [])

    def get_search_terms_by_profile(self, profile: str) -> List[str]:
        """Get search terms for a specific professional profile."""
        return self.professional_profiles.get(profile, [])

    def generate_combined_search_terms(self, profile: str, pain_point: str) -> List[str]:
        """Generate combined search terms for a profile and pain point."""
        profiles = self.get_search_terms_by_profile(profile)
        pain_points = self.get_search_terms_by_pain_point(pain_point)
        
        combined_terms = []
        for prof in profiles:
            for pain in pain_points:
                combined_terms.append(f"{prof} {pain}")
        
        return combined_terms

    def get_all_categories(self) -> Dict[str, List[str]]:
        """Get all available categories for pain points and profiles."""
        return {
            "pain_points": list(self.business_pain_points.keys()),
            "profiles": list(self.professional_profiles.keys())
        }

# Example usage
if __name__ == "__main__":
    tool = CustomerProfileSearchTool()
    
    # Print available categories
    print("Available categories:")
    print(json.dumps(tool.get_all_categories(), indent=2))
    
    # Example: Generate combined search terms for business owner with productivity pain point
    combined_terms = tool.generate_combined_search_terms("business_owner", "productivity")
    print("\nExample combined search terms:")
    for term in combined_terms[:3]:  # Show first 3 examples
        print(f"- {term}") 