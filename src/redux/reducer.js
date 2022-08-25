const initState = {
    people: [
        {name: 'Duc Hieu', phone: '0932120098', email: 'duchieu@gmail.com'}
    ]
}
const rootReducer = (state = initState, action) => {
    switch(action.type){
        case 'addCard':{
            return{
                ...state,
                people: [
                    ...state.people,
                    action.payload
                ]
            };
        } 
        case 'deleteCard':{
            let newList = state.people;
            newList = newList.filter((item, i) => i != action.atIndex);
            return{
                ...state,
                people: newList
            };
        }
        case 'editCard':{
            let newList = [...state.people];
            newList[action.atIndex] = action.payload
            return{
                ...state,
                people: newList
            };
        }
            
        default:
            return state
    }
}

export default rootReducer