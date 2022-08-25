import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';

const AddButton = (props) => {
    const name = '+';
    return(
        <TouchableOpacity
            onPress={props.navigation}
            style = {styles.container}
            activeOpacity = {0.7}>
                <View>
                    <Text style = {styles.context}>
                        {name}
                    </Text>
                </View>
        </TouchableOpacity>           
    );
}

const styles = StyleSheet.create({
    container:{
        width: 50,
        height:50,
        borderRadius: 50,
        justifyContent: 'center',
        alignItems: 'center',
        alignSelf:'center',
        backgroundColor: '#26c6da',
        bottom: 15,
    },
    context:{
        fontSize: 30,
        color: '#FFF'
    },
});

export default AddButton;