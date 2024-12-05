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
