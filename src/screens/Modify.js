import React from 'react';
import { useState } from 'react';

import { 
    StyleSheet, 
    View, 
    TouchableOpacity, 
    Text 
} from 'react-native';

import { useSelector, useDispatch } from 'react-redux';
import {editCardAction} from '../redux/actions.js'
import {addCardAction} from '../redux/actions.js'

import InformationSection from '../components/InformationSection.js';

const Modify = ({route, navigation}) => {
    const iPerson = route.params.i 
    const isAdd = route.params.isAdd

    const buttonName = (isAdd ? 'Submit' : 'Save changes')
    const info = useSelector((state) => state.people[iPerson])

    const dispatch = useDispatch()
    const handleError = (text, input) => {
        setError(previousState => ({...previousState, [input]: text}))
    }
    const changeCardOnPress = () => {
        let valid = true
        if(!inputName) {
            handleError('Please input your name', 'name')
            valid = false
        }
        else {
            handleError('', 'name')
            valid = true
        }
        if(!inputPhone) {
            handleError('Please input your phone number', 'phone')
            valid = false
        }
        else {
            handleError('', 'phone')
            valid = true
        }
        if(!inputEmail) {
            handleError('Please input your email', 'email')
            valid = false
        }
        else if(!inputEmail.match(/\S+@\S+\.\S+/)){
            handleError('Please input valid email', 'email')
            valid = false
        }
        else {
            handleError('', 'email')
            valid = true
        }
        if(valid === true) {
            if(isAdd){
                dispatch(addCardAction({
                    name: inputName,
                    phone: inputPhone,
                    email: inputEmail
                  }))
                navigation.navigate('Home')
            }
            else{
                dispatch(editCardAction(iPerson,
                    {
                    name: inputName,
                    phone: inputPhone,
                    email: inputEmail
                  }))
            }
            navigation.navigate('Home')
        }
}
    const [inputName, setName] = useState((isAdd ? '' : info.name))
    const [inputPhone, setPhone] = useState((isAdd ? '' : info.phone))
    const [inputEmail, setEmail] = useState((isAdd ? '' : info.email))
    const [errorMsg, setError] = useState({})

    return(
        <View>
            <InformationSection label ={(isAdd ? "Enter your name" : "Name")} 
            defaultValue = {(isAdd ? "" : info.name)}
            onChangeText={newText => setName(newText)}
            autoCapitalize = "words" 
            error={errorMsg.name}
            />

            <InformationSection label = {(isAdd ? "Enter your phone number" :"Phone Number" )}
            defaultValue = {(isAdd ? "" : info.phone)}
            onChangeText={newText => setPhone(newText)}
            keyboardType="numeric" 
            error={errorMsg.phone}
            />

            <InformationSection label = {(isAdd ? "Enter your email" :"Email" )}
            defaultValue = {(isAdd ? "" : info.email)}
            onChangeText={newText => setEmail(newText)} 
            error = {errorMsg.email}
            />

            <TouchableOpacity onPress={changeCardOnPress}
                style = {styles.buttonWrapper}
                activeOpacity = {0.7}>
                    <Text style = {styles.button}>
                        {buttonName}
                    </Text>
            </TouchableOpacity>     
        </View>
    )
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: '#e3f2fd',
        flexDirection: 'column',
    },
    buttonWrapper:{
        alignSelf: 'center',
        alignItems:'center',
        justifyContent:'center',
        backgroundColor: '#303f9f',
        width: 200,
        height: 40,
        marginVertical: 10
    },
    button: {
        color: '#FFF',
        fontWeight: 'bold',
        fontSize: 20,
    },
    input:{
        margin: 5,
        height: 40,
        borderColor: 'gray',
        borderWidth: 1,
        width: 350,
        alignSelf: 'center'
    },
    text: {
        marginLeft: 20,
        marginBottom:10,
        marginTop: 10,
    }
})

export default Modify;

