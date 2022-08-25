import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import {Provider} from 'react-redux';
// SCREEN
import Home from './src/screens/Home';
import Modify from './src/screens/Modify';
import {store, persistor} from './src/redux/store';

import { PersistGate } from 'redux-persist/integration/react'
const Stack = createNativeStackNavigator();

const App = () => {
  return (
    <Provider store={store}>
      <PersistGate loading={null} persistor={persistor}>

        <NavigationContainer>
          <Stack.Navigator initialRouteName='Home'>
            <Stack.Screen name="Home" 
              component={Home} 
              options ={{
                title: 'Home',
                headerTitleAlign: "center",
                headerStyle: {
                  backgroundColor: '#0288d1',
                },
                headerTitleStyle: {
                  color: "#FFF",
                  fontWeight: "bold",
                }
              }}/>
              <Stack.Screen name="Modify" 
              component={Modify} 
              options ={{
                title: 'Modify',
                headerTitleAlign: "center",
                headerStyle: {
                  backgroundColor: '#0288d1',
                },
                headerTitleStyle: {
                  color: '#FFF',
                  fontWeight: "bold",
                }
              }}
              />
          </Stack.Navigator>
        </NavigationContainer>
      
      </PersistGate>
    </Provider>
  );
}
export default App;