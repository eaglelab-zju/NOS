The limitations of 16 existing UI tasks for addressing NA, OA and SA issues are demonstrated in table bellow.

|  | Existing UI Tasks and Their Limitation for Accessibility                                                                                         | N. | O. | S. |
|------|--------------------------------------------------------------------------------------------------------------------------------------------------|----|----|----|
| 1    | **UI Detection**: Identifying visible components without assessing accessibility requirements or focusability.                                      | ✗  | ✗  | ✗  |
| 2    | **Type Recognition**: Only classifies component types; does not determine focusability accessibility needs.                                         | ✗  | ✗  | ✗  |
| 3    | **Function Inference**: Infers potential functionality but lacks granularity to decide on selective exposure for AT.                                | ✗  | ✗  | ✗  |
| 4    | **Tappability Prediction**: Identifies clickable elements; does not address focusability for non-clickable but accessible components.               | ✗  | ✗  | ✗  |
| 5    | **Widget Captioning**: Generates captions for individual components but does not handle structural or hierarchical focusability.                    | ✗  | ✗  | ✗  |
| 6    | **Screen Summarization**: Provides an overview of the screen without focusable element distinction needed for accessibility.                        | ✗  | ✗  | ✗  |
| 7    | **UI Component Suggestion**: Suggests components but lacks predictions on whether a component should be focusable.                                  | ✗  | ✗  | ✗  |
| 8    | **Touch Gesture Recognition**: Focuses on user gestures without addressing accessibility requirements.                                              | ✗  | ✗  | ✗  |
| 9    | **User Intent Detection**: Detects general intent but does not provide context-sensitive accessibility predictions.                                 | ✗  | ✗  | ✗  |
| 10   | **User Flow Prediction**: Predicts user flows but does not consider selective focusability or component access needs.                               | ✗  | ✗  | ✗  |
| 11   | **Screen Transition Prediction**: Forecasts screen changes without attention to individual element accessibility.                                   | ✗  | ✗  | ✗  |
| 12   | **UI Aesthetics Evaluation**: Evaluates visual appeal without considering accessibility or focusability for users with disabilities.                | ✗  | ✗  | ✗  |
| 13   | **Command Grounding**: Maps commands to actions but does not predict context-based accessibility of components.                                     | ✗  | ✗  | ✗  |
| 14   | **Interaction Modeling**: Models multi-modal interactions yet lacks predictive accessibility within complex interfaces.                             | ✗  | ✗  | ✗  |
| 15   | **Conversation Perception**: Interprets conversational cues without addressing focusability and accessibility within the UI.                        | ✗  | ✗  | ✗  |
| 16   | **Conversation Interaction**: Facilitates conversation-based interactions but does not predict component accessibility needs.                       | ✗  | ✗  | ✗  |
|    | **UI Focusability Prediction (UFP)**: Achieve unified predictions on component focusability and granularity.                                    | ✓  | ✓  | ✓  |

## UI Function-targeted Tasks
1. **UI Detection**  
   Detecting and segmenting UI components in screen images, which aids in tasks like automated testing or accessibility enhancement.

2. **Type Recognition**  
   Identifying the functional type of UI components (e.g., button, text field) based on visual or structural features.

3. **Function Inference**  
   Inferring functionality of UI elements to facilitate automated interactions, especially in complex or dynamic interfaces.

4. **Tappability**  
   Assessing whether UI elements are easily tappable, especially on touch devices, considering size, placement, and accessibility.

## UI Description Generation Tasks
1. **Widget Captioning/Detailed Description Generation**  
   Generating descriptive captions or comprehensive descriptions for UI elements (widgets) and layouts to improve accessibility and assistive technologies.

2. **Screen Summarization**  
   Condensing the content and layout of a screen to provide a quick overview, which is particularly useful for users with visual impairments.

## User Behavior-targeted Tasks
1. **UI Component Suggestion**  
   Automatically suggesting UI elements based on content and context, facilitating user-centered design.

2. **Touch Gesture Recognition**  
   Recognizing multi-touch or gesture patterns and linking them to UI functions for a smoother, gesture-based interaction.

3. **User Intent Detection**  
   Identifying user intent based on interactions with UI elements, which can enhance responsiveness and adaptability in applications.

4. **User Flow Prediction**  
   Predicting user navigation patterns or next actions within an interface based on historical or contextual data, improving UX by personalizing pathways.

5. **Screen Transition Prediction**  
   Modeling likely transitions between screens in an app to streamline workflows and support user needs.

6. **UI Aesthetics Evaluation**  
   Assessing and enhancing UI aesthetics through computational models that predict aesthetic quality and user satisfaction.

## FM-based Hybrid Tasks
1. **Command Grounding**  
   Mapping user commands (e.g., voice or text inputs) to corresponding UI actions or elements. Techniques using natural language processing and UI context have been proposed recently.

2. **Multi-modal Interaction Modeling**  
   Modeling interactions across different modalities (touch, voice, visual) for consistent experiences across devices.

3. **Conversation Perception**  
   Understanding contextual user conversations around UI elements, assisting in adaptive interface updates.

4. **Conversation Interaction**  
   Analyzing interaction patterns in conversational interfaces to enhance engagement and usability.
