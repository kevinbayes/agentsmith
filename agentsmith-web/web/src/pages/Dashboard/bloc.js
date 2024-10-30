import {BaseBloc} from "../../components/BaseComponent/bloc.js";


export class Bloc extends BaseBloc {

  constructor(options) {
    super(options);
  }

  setQuery(query) {
    this.__updateSubject({query: query});
  }

  runQuery() {

  }

}

export class Event {

}
