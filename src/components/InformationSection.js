import React from 'react';
import { View, Text, StyleSheet, TextInput } from 'react-native';

const InformationSection = ( {
    label,
    error,
    ...props
}) => {
    return(
       
        <View>
                <Text style = {styles.text}>
                    {label}
                </Text>
                <TextInput 
                style={[styles.input, {borderColor: error ? '#ff1744' : 'gray',}]}
                {...props}
                />
                {error && (
                <Text style = {[{color: '#ff1744', fontWeight: "bold", marginTop: 0, marginBottom: 0},styles.text]}>
                {error}
                </Text>
                )}
        </View>   
    );
}

const styles = StyleSheet.create({
    text: {
        marginLeft: 22,
        marginBottom:10,
        marginTop: 10,
    },
    input:{
        margin: 5,
        height: 40,
        borderWidth: 1,
        width: 350,
        alignSelf: 'center'
    },
});

export default InformationSection;