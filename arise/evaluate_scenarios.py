import os
import re
import random
import datetime
from cv2 import accumulate
from google import genai
from google.genai import types

with open('eval_prompt.txt', 'r') as file:
    eval_prompt = file.read()

with open('list_of_scenarios.txt', 'r') as file:
    list_of_scenarios = file.read().splitlines()

REF_SCENARIOS_FOLDER = 'reference_scenarios/'
SCENARIOS_FOLDER = 'eval_test_set/gemini'

scenario_folders = [f for f in os.listdir(
    SCENARIOS_FOLDER) if os.path.isdir(os.path.join(SCENARIOS_FOLDER, f))]

def generate(message):
    client = genai.Client(
        api_key="YOUR_API_KEY",
    )

    model = "gemini-2.5-flash"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="""Here is a shortened version of scenic 3.0 documentation that is relevant to the evaluation of scenarios:

                    Scenic is a domain-specific probabilistic programming language designed for modeling and generating scenarios for cyber-physical systems, particularly in robotics and autonomous driving. It allows users to define distributions over scenes (static configurations of objects) and dynamic policies (how agents act over time), enabling the generation of diverse training and testing data for simulators.

                    **Key takeaway for version 3.0:** It introduces native **3D geometry**, precise **object shapes**, and enhanced **temporal requirements**, with significant syntax and semantic changes from previous versions (e.g., requiring the `new` keyword for object instantiation, and changes to `heading` property derivation).

                    ---

                    Scenic's syntax is inspired by Python, but extends it with specialized keywords and operators for scenario definition.

                    *   **Object Instantiation:**
                        *   The `new` keyword is now **required** for creating instances of classes (e.g., `ego = new Object`). This resolves ambiguity from earlier versions.
                        *   Objects are defined with a set of `specifiers` (see Spatial Relations below).
                            ```scenic
                            ego = new Object with shape ConeShape(),
                                        with width 2,
                                        with height 1.5,
                                        facing (-90 deg, 45 deg, 0)
                            ```

                    *   **Class Definitions:**
                        *   Similar to Python classes, supporting inheritance and methods.
                        *   Properties can be defined with default values, which can depend on other properties of `self`.
                            ```scenic
                            class Vehicle:
                                pass

                            class Car:
                                position: new Point on road
                                heading: roadDirection at self.position
                                width: self.model.width
                            ```

                    *   **Distributions:** Scenic is a probabilistic language, so random values are defined using distributions.
                        *   `Range(low, high)`: Uniformly-distributed real number.
                        *   `DiscreteRange(low, high)`: Uniformly-distributed integer.
                        *   `Normal(mean, stdDev)`: Normal distribution.
                        *   `TruncatedNormal(mean, stdDev, low, high)`: Truncated normal distribution.
                        *   `Uniform(value1, value2, ...)`: Uniformly selects from a finite set of values.
                        *   `Discrete({value: weight, ...})`: Discrete distribution with given weights.

                    *   **Requirements/Constraints (`require`):**
                        *   Define conditions that must be satisfied by generated scenes or simulations.
                        *   **Hard Requirements:** `require boolean_expression` (e.g., `require car2 can see ego`). If violated, the scene/simulation is rejected.
                        *   **Soft Requirements:** `require[probability] boolean_expression` (e.g., `require[0.5] car2 can see ego`).
                        *   **Temporal Requirements:** Use temporal operators within `require` for dynamic scenarios:
                            *   `always condition`
                            *   `eventually condition`
                            *   `next condition`
                            *   `condition until condition`
                            *   `hypothesis implies conclusion`

                    *   **Behaviors (`behavior`):** Define dynamic policies for agents.
                        *   Functions that execute actions over time using `take action` or `wait`.
                        *   Can have `precondition` and `invariant` guards.
                        *   Control flow (`if`, `while`) inside behaviors can depend on random values (unlike top-level code).
                            ```scenic
                            behavior FollowLaneBehavior():
                                while True:

                                    take SetThrottleAction(throttle), SetSteerAction(steering)
                            ```

                    *   **Monitors (`monitor`):** Run in parallel with scenarios, typically for checking properties without taking actions.
                        *   Can maintain state (local variables).
                        *   Instantiated using `require monitor monitor_name(args)`.

                    *   **Modular Scenarios (`scenario`):** Define reusable scenario components.
                        *   `setup:` block: executed once at compilation, defines objects, requirements.
                        *   `compose:` block: orchestrates execution of other scenarios/behaviors over time using `do`.
                        *   Composition statements: `do scenario`, `do scenario until condition`, `do scenario for N seconds/steps`, `do choose scenario1, scenario2`, `do shuffle scenario1, scenario2`.

                    *   **Mutations (`mutate`):** Randomly vary properties of existing objects for testing.
                        *   `mutate object_list [by scalar_factor]`

                    *   **Recording (`record`):** Save values during simulation for analysis.
                        *   `record value as name` (at every time step).
                        *   `record initial value as name` (at start).
                        *   `record final value as name` (at end).

                    *   **Interrupts (`try: ... interrupt when ...`):** Allows behaviors/scenarios to suspend and execute an interrupt handler when a condition is met. `abort` can terminate the `try-interrupt` block.

                    ---

                    Scenic programs define a **probability distribution** over possible scenes and dynamic evolutions. The overall process involves:

                    1.  **Compilation:** A Scenic program is compiled into a `Scenario` object. This involves parsing the Scenic code into an Abstract Syntax Tree (AST), transforming it into a Python AST, and executing it. During this phase, distributions and requirements are set up.
                    2.  **Scene Generation (Sampling):** From the `Scenario` object, concrete `Scene` objects are sampled. This is a rejection sampling process: Scenic makes random choices (from distributions) and then checks if the resulting scene satisfies all *hard requirements*. If not, the sample is rejected, and a new one is tried. Pruning techniques help make this more efficient by avoiding infeasible parts of the sample space.
                        *   A `Scene` object contains: `objects` (physical objects, including `egoObject`), `params` (global parameters), and `workspace` (the overall bounding region).
                    3.  **Dynamic Simulation (Execution):** For dynamic scenarios, a `Simulator` is used to run the `Scene` over time. The behaviors of agents run in parallel, taking actions at discrete time steps. Requirements (including temporal ones and monitor checks) are continuously evaluated, and violations lead to rejection of the simulation.

                    **Hierarchical Object Model:**
                    Scenic provides a built-in hierarchy for defining objects:
                    *   **`Point`**: Basic spatial location (3D coordinates: `(x, y, z)`). Default Z is 0 for 2D compatibility.
                    *   **`OrientedPoint`**: Extends `Point` with an `orientation`. Orientation is a 3D quaternion, derived from `parentOrientation` and intrinsic `yaw`, `pitch`, `roll` Euler angles.
                    *   **`Object`**: Extends `OrientedPoint` with physical properties:
                        *   `width`, `length`, `height` (dimensions of its bounding box).
                        *   `shape` (e.g., `BoxShape`, `ConeShape`, `MeshShape` loaded from STL files).
                        *   `allowCollisions` (bool, default `False`: objects don't overlap).
                        *   `regionContainedIn` (the `Region` the object must be within, default `workspace`).
                        *   `requireVisible` (bool, default `False`: object must be visible from `ego`).
                        *   `behavior` (dynamic policy for agents).
                        *   `velocity`, `speed`, `angularVelocity`, `angularSpeed` (dynamic state).

                    ---

                    Spatial relations are a core strength of Scenic, allowing for natural language-like descriptions of how objects are positioned relative to each other and the environment. This is achieved through powerful **specifiers** and **operators**.

                    These are used after `new ClassName` to define its properties. They often interact with each other and are resolved by Scenic using a priority system.

                    *   **Absolute Positioning:**
                        *   `at vector`: Places the object at global coordinates.
                            ```scenic
                            new Object at (10, 5, 2)
                            ```
                    *   **Relative Positioning (to `ego` by default, or another object/point):**
                        *   `offset by vector`: Position relative to the `ego`'s local coordinate system.
                            ```scenic
                            new Object offset by (5, 0, 0)
                            ```
                        *   `offset along direction by vector`: Position relative to a given direction.
                        *   `(left | right) of (vector | OrientedPoint | Object) [by scalar]`: Positions to the left/right. The `by` scalar defines the distance between bounding boxes.
                            ```scenic
                            new Car left of ego by 2
                            ```
                        *   `(ahead of | behind) (vector | OrientedPoint | Object) [by scalar]`: Positions ahead/behind.
                        *   `(above | below) (vector | OrientedPoint | Object) [by scalar]`: Positions above/below.
                        *   `following vectorField [from vector] for scalar`: Positions by following a vector field (e.g., a road).

                    *   **Region-Based Positioning:** Scenic's `Region` objects are fundamental for spatial constraints.
                        *   `in region`: Places object uniformly at random *within* a specified `Region`.
                            ```scenic
                            new Pedestrian in sidewalkRegion
                            ```
                        *   `on (region | Object | vector)`: Places the *base* of the object uniformly at random on a surface (e.g., `on floor`, `on road`). This specifier also *modifies* existing positions by projecting them onto the region/surface.
                            ```scenic
                            new Rock on MarsGround
                            ```
                        *   `contained in region`: Ensures the *entire object* (not just center/base) is contained within a region.

                    *   **Orientation Specifiers:**
                        *   `facing orientation`: Sets the object's global orientation directly.
                        *   `facing vectorField`: Orients the object along the direction of a vector field at its position (e.g., `facing roadDirection`).
                        *   `facing (toward | away from) vector`: Orients the object to face toward/away from a given point.
                        *   `facing directly (toward | away from) vector`: Sets both yaw and pitch to face a point.
                        *   `apparently facing heading [from vector]`: Orients the object based on its apparent heading relative to `ego`'s line of sight.

                    These are functions that return values based on spatial relationships between objects or points.

                    *   **Scalar Operators (distances, angles):**
                        *   `distance [from vector] to vector`: Euclidean distance.
                        *   `angle [from vector] to vector`: Azimuthal angle.
                        *   `altitude [from vector] to vector`: Vertical distance.
                        *   `relative heading of heading [from heading]`: Relative heading.
                        *   `apparent heading of OrientedPoint [from vector]`: Apparent heading from `ego`.

                    *   **Boolean Operators (conditions):**
                        *   `(Point | OrientedPoint | Object) can see (vector | Object)`: Checks visibility, **accounting for occlusion and 3D shapes** (a major 3.0 feature).
                        *   `(vector | Object) in region`: Checks if a point/object is contained within a region.
                        *   `(Object | region) intersects (Object | region)`: Checks for overlap between shapes/regions.

                    *   **Region Operators (new regions):**
                        *   `visible region`: Returns the portion of a region visible from `ego` (or other point).
                        *   `not visible region`: Returns the portion not visible.

                    *   **OrientedPoint Operators (accessing object parts):**
                        *   `(front | back | left | right | top | bottom) of Object`: Returns an `OrientedPoint` at the midpoint of a specific side of an object's bounding box.
                        *   Combinations like `TopFrontLeft of Object`.

                    ---
