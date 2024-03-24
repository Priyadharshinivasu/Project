import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { User } from './user.model';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class UserService {

  constructor(private httpClient:HttpClient) { }

  apiURL = "http://localhost:8080/ShoppingMall"

  createUser(user:User):Observable<User>
  {
    return this.httpClient.post<User>(this.apiURL,user);
  }

  getUser():Observable<User[]>
  {
    return this.httpClient.get<User[]>(this.apiURL);
  }

  updateUser(id:number,user:User):Observable<User>
  {
      return this.httpClient.put<User>(this.apiURL+'/'+id,user);
  }

  deleteUser(id:number)
  {
      return this.httpClient.delete(this.apiURL+'/'+id);
  }

}
