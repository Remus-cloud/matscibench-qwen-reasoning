import ast
import math
from dataclasses import dataclass
from pint.errors import PintError
import quantity_calculator as qc

@dataclass(frozen=True)
class Vector:
    components: tuple

class Extensions(ast.NodeTransformer):
    """Translate a whitelist of expressions into typed local calculator bindings."""
    def __init__(self,variables):
        self.variables=variables
        self.count=0

    def bind(self,obj):
        self.count+=1
        name=f"compatLocal{self.count}"
        self.variables[name]=obj
        return ast.Name(id=name,ctx=ast.Load())

    def evaluate(self,node):
        return qc.value(self.visit(node),self.variables)

    def visit_List(self,node):
        if not 1<=len(node.elts)<=12:
            raise ValueError("Unsupported vector length")
        return self.bind(Vector(tuple(self.evaluate(e) for e in node.elts)))

    visit_Tuple=visit_List

    def visit_Subscript(self,node):
        vector=self.evaluate(node.value)
        if not isinstance(vector,Vector) or not isinstance(node.slice,ast.Constant) or type(node.slice.value)!=int:
            raise ValueError("Expected a fixed vector and literal component index")
        index=node.slice.value
        if not 0<=index<len(vector.components):
            raise ValueError("Vector component outside range")
        return self.bind(vector.components[index])

    def visit_Attribute(self,node):
        if node.attr!="magnitude":
            raise ValueError("Only explicit unit conversion and magnitude are supported")
        q=self.evaluate(node.value)
        if not isinstance(q,qc.UNITS.Quantity):
            raise ValueError("Magnitude requires a quantity")
        return self.bind(float(q.magnitude))

    def visit_Call(self,node):
        if isinstance(node.func,ast.Attribute) and node.func.attr=="to" and not node.keywords and len(node.args)==1:
            unit=node.args[0]
            if not isinstance(unit,ast.Constant) or type(unit.value)!=str:
                raise ValueError("Expected a literal unit name")
            q=self.evaluate(node.func.value)
            return self.bind(qc.UNITS.Quantity(q).to(qc.as_unit(unit.value)))
        if isinstance(node.func,ast.Name) and node.func.id=="Q" and not node.keywords and len(node.args) in (1,2):
            val=self.evaluate(node.args[0])
            if isinstance(val,Vector):
                raise ValueError("Expected scalar quantity")
            if len(node.args)==1:
                return self.bind(qc.UNITS.Quantity(val))
            unit=node.args[1]
            if not isinstance(unit,ast.Constant) or type(unit.value)!=str:
                raise ValueError("Expected a literal unit name")
            dest=qc.as_unit(unit.value)
            q=val.to(dest) if isinstance(val,qc.UNITS.Quantity) else qc.UNITS.Quantity(val,dest)
            return self.bind(q)
        return self.generic_visit(node)

def calculate(worksheet,target):
    try:
        return qc.calculate(worksheet,target)
    except ValueError:
        pass
    if len(worksheet)>5000:
        raise ValueError("Worksheet too long")
    try:
        tree=ast.parse(worksheet.strip().replace("^","**"))
        nodes=list(ast.walk(tree))
        if len(nodes)>500 or not tree.body or not isinstance(tree.body[-1],ast.Expr):
            raise ValueError("Expected bounded numerical worksheet")
        if any(isinstance(n,ast.Name) and n.id.startswith("compatLocal") for n in nodes):
            raise ValueError("Reserved local variable")
        variables={"pi":math.pi}
        visitor=Extensions(variables)
        for stmt in tree.body[:-1]:
            if not isinstance(stmt,ast.Assign) or len(stmt.targets)!=1 or not isinstance(stmt.targets[0],ast.Name):
                raise ValueError("Only plain assignments allowed")
            name=stmt.targets[0].id
            if name.startswith("_"):
                raise ValueError("Invalid variable name")
            variables[name]=visitor.evaluate(stmt.value)
        q=visitor.evaluate(tree.body[-1].value)
        val=float(qc.UNITS.Quantity(q).to(qc.as_unit(target)).magnitude)
        if not math.isfinite(val):
            raise ValueError("Nonfinite result")
        return val
    except (PintError,ValueError,TypeError,ArithmeticError,SyntaxError,RecursionError) as e:
        raise ValueError(str(e)) from None

