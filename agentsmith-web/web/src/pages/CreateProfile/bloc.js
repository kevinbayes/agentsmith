import {BaseBloc} from "../../components/BaseComponent/bloc.js";

export class Bloc extends BaseBloc {

  globalBloc;

  constructor(options, globalBloc) {
    super(options);
    this.globalBloc = globalBloc;
  }

  initialise = () => {

    const { user }  = this.subject.value;

    this.__makeInitialised(
        { form: { given_name: user.given_name, family_name: user.family_name, email: user.email, email_consent: false, terms: false }}
    );
  }

  text_changed = (e) => {

    let form = this.subject.value.form;
    form[e.target.name] = e.target.value;
    this.__updateSubject({ form })
  }

  checkbox_changed = (e) => {

    let form = this.subject.value.form;
    form[e.target.name] = e.target.checked;
    this.__updateSubject({ form })
  }

  submit = (e) => {
    e.preventDefault();
    this.__updateSubject({ error: undefined, processing: true });
    let form = this.subject.value.form;
    this.logger.debug("Submitting form: ", form);
    this.globalBloc.stores().accountStore.createProfile(form)
      .then(value => {
        if(value.status > 299) {
          this.__updateSubject({ error: "Unable to create profile. Please try again and if the problem persists contact support." })
        } else {
          this.__updateSubject({ success: true })
          this.globalBloc.setProfile(value.data);
        }
      }, error => {
        this.logger.error("Error: ", error);
        this.__updateSubject({ error: "Unable to create profile. Please try again and if the problem persists contact support." })
      }).finally(() => {
        this.__updateSubject({ processing: false });
      });
  }
}

export class Event {

}
